# cspell:disable
import logging
from copy import deepcopy
from typing import List, Union
from typing import Tuple
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm
import numpy as np
import scipy.stats
from discrete_optimization.generic_tools.graph_api import Graph
from discrete_optimization.rcpsp.rcpsp_model import RCPSPModel, RCPSPSolution
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon as pp
from shapely.geometry import Polygon

import webcolors

css_colors = list(webcolors.CSS3_NAMES_TO_HEX.values())

plt.style.use('seaborn-dark')


logger = logging.getLogger(__name__)


def compute_resource_consumption(
    rcpsp_model: RCPSPModel,
    rcpsp_sol: RCPSPSolution,
    list_resources: List[Union[int, str]] = None,
    future_view=True,
):
    modes_extended = deepcopy(rcpsp_sol.rcpsp_modes)
    modes_extended.insert(0, 1)
    modes_extended.append(1)
    modes_dict = rcpsp_model.build_mode_dict(rcpsp_sol.rcpsp_modes)
    last_activity = rcpsp_model.sink_task
    makespan = rcpsp_sol.rcpsp_schedule[last_activity]["end_time"]
    if list_resources is None:
        list_resources = rcpsp_model.resources_list
    consumptions = np.zeros((len(list_resources), makespan + 1))
    for act_id in rcpsp_sol.rcpsp_schedule:
        for ir in range(len(list_resources)):
            use_ir = rcpsp_model.mode_details[act_id][modes_dict[act_id]].get(
                list_resources[ir], 0
            )
            if future_view:
                consumptions[
                    ir,
                    rcpsp_sol.rcpsp_schedule[act_id]["start_time"]
                    + 1: rcpsp_sol.rcpsp_schedule[act_id]["end_time"]
                    + 1,
                ] += use_ir
            else:
                consumptions[
                    ir,
                    rcpsp_sol.rcpsp_schedule[act_id][
                        "start_time"
                    ]: rcpsp_sol.rcpsp_schedule[act_id]["end_time"],
                ] += use_ir

    return consumptions, np.arange(0, makespan + 1, 1)


def compute_nice_resource_consumption(
    rcpsp_model: RCPSPModel,
    rcpsp_sol: RCPSPSolution,
    list_resources: List[Union[int, str]] = None,
):
    if list_resources is None:
        list_resources = rcpsp_model.resources_list
    c_future, times = compute_resource_consumption(
        rcpsp_model, rcpsp_sol, list_resources=list_resources, future_view=True
    )
    c_past, times = compute_resource_consumption(
        rcpsp_model, rcpsp_sol, list_resources=list_resources, future_view=False
    )
    merged_times = {i: [] for i in range(len(list_resources))}
    merged_cons = {i: [] for i in range(len(list_resources))}
    for r in range(len(list_resources)):
        for index_t in range(len(times)):
            merged_times[r] += [times[index_t], times[index_t]]
            merged_cons[r] += [c_future[r, index_t], c_past[r, index_t]]
    for r in merged_times:
        merged_times[r] = np.array(merged_times[r])
        merged_cons[r] = np.array(merged_cons[r])
    return merged_times, merged_cons


def plot_ressource_view(
    rcpsp_model: RCPSPModel,
    rcpsp_sol: RCPSPSolution,
    list_resource: List[Union[int, str]] = None,
    title_figure="",
    x_lim=None,
    fig=None,
    ax=None,
):
    modes_extended = deepcopy(rcpsp_sol.rcpsp_modes)
    modes_extended.insert(0, 1)
    modes_extended.append(1)
    with_calendar = rcpsp_model.is_varying_resource()
    modes_dict = rcpsp_model.build_mode_dict(rcpsp_sol.rcpsp_modes)
    tasks = rcpsp_model.tasks_list
    nb_task = len(tasks)
    if list_resource is None:
        list_resource = rcpsp_model.resources_list
    if ax is None:
        fig, ax = plt.subplots(nrows=len(list_resource), figsize=(20, 10), sharex=True)
        fig.suptitle(title_figure)
    polygons_ax = {i: [] for i in range(len(list_resource))}
    labels_ax = {i: [] for i in range(len(list_resource))}
    sorted_activities = sorted(
        rcpsp_sol.rcpsp_schedule,
        key=lambda x: rcpsp_sol.rcpsp_schedule[x]["start_time"],
    )
    nb_colors = rcpsp_model.mode_details[nb_task][1]['org_task_id'] // 2
    colors = plt.cm.get_cmap("hsv", nb_colors)
    for j in sorted_activities:
        time_start = rcpsp_sol.rcpsp_schedule[j]["start_time"]
        time_end = rcpsp_sol.rcpsp_schedule[j]["end_time"]
        for i in range(len(list_resource)):
            cons = rcpsp_model.mode_details[j][modes_dict[j]].get(list_resource[i], 0)
            if cons == 0:
                continue
            bound = (
                rcpsp_model.resources[list_resource[i]]
                if not with_calendar
                else max(rcpsp_model.resources[list_resource[i]])
            )
            for k in range(0, bound):
                polygon = Polygon(
                    [
                        (time_start, k),
                        (time_end, k),
                        (time_end, k + cons),
                        (time_start, k + cons),
                        (time_start, k),
                    ]
                )
                areas = [p.intersection(polygon).area for p in polygons_ax[i]]
                if len(areas) == 0 or max(areas) == 0:
                    polygons_ax[i].append(polygon)
                    labels_ax[i].append(j)
                    break
    for i in range(len(list_resource)):
        patches = []
        for index, polygon in enumerate(polygons_ax[i]):
            x, y = polygon.exterior.xy
            pos = labels_ax[i][index]
            pos = rcpsp_model.mode_details[pos][1]['org_task_id'] - 1
            ax[i].plot(x, y, zorder=-1, color=colors((pos) % nb_colors))
            patches.append(pp(xy=polygon.exterior.coords, facecolor=colors((pos) % nb_colors)))
        p = PatchCollection(patches, match_original=True, alpha=0.4)
        ax[i].add_collection(p)
    merged_times, merged_cons = compute_nice_resource_consumption(
        rcpsp_model, rcpsp_sol, list_resources=list_resource
    )
    for i in range(len(list_resource)):
        ax[i].plot(
            merged_times[i],
            merged_cons[i],
            color="r",
            linewidth=2,
            label="Consumption " + str(list_resource[i]),
            zorder=1,
        )
        if not with_calendar:
            ax[i].axhline(
                y=rcpsp_model.resources[list_resource[i]],
                linestyle="--",
                label="Limit : " + str(list_resource[i]),
                zorder=0,
            )
        else:
            merged_disp = []
            value = rcpsp_model.resources[list_resource[i]][0]
            merged_disp.append(value)
            merged_disp.append(value)
            for index, m in enumerate(merged_times[i][:-2]):
                value = rcpsp_model.resources[list_resource[i]][m + 1 if int(index % 2) else m]
                merged_disp.append(value)

            ax[i].plot(
                merged_times[i],
                merged_disp,
                linestyle="--",
                label="Limit : " + str(list_resource[i]),
                zorder=0,
            )
        ax[i].legend(fontsize=5)
        lims = ax[i].get_xlim()
        if x_lim is None:
            ax[i].set_xlim([lims[0], 1.0 * lims[1]])
        else:
            ax[i].set_xlim(x_lim)
    ax[-1].set_xlabel("Sprint")
    return fig


def con_style(ax, color,
              org_pos_x, org_pos_y,
              dst_pos_x, dst_pos_y,
              connectionstyle):
    x1, y1 = dst_pos_x, dst_pos_y
    x2, y2 = org_pos_x, org_pos_y

    ax.plot([x1, x2], [y1, y2], ".", color=color)
    ax.annotate("",
                xy=(x1, y1), xycoords='data',
                xytext=(x2, y2), textcoords='data',
                arrowprops=dict(arrowstyle="->", color=color,
                                linewidth=1,
                                shrinkA=5, shrinkB=5,
                                patchA=None, patchB=None,
                                connectionstyle=connectionstyle,
                                ),
                )


def plot_task_gantt(
    rcpsp_model: RCPSPModel,
    rcpsp_sol: RCPSPSolution,
    fig=None,
    ax=None,
    x_lim=None,
    title=None,
    current_t=None,
):
    if fig is None or ax is None:
        fig, ax = plt.subplots(1, figsize=(20, 10))
        ax.set_title("Gantt Task")
    if title is None:
        ax.set_title("Gantt Task")
    else:
        ax.set_title(title)
    tasks = rcpsp_model.tasks_list
    nb_task = len(tasks)
    sorted_task_by_start = sorted(
        rcpsp_sol.rcpsp_schedule,
        key=lambda x: 100000 * rcpsp_sol.get_start_time(x) + rcpsp_model.index_task[x],
    )
    sorted_task_by_end = sorted(
        rcpsp_sol.rcpsp_schedule,
        key=lambda x: 100000 * rcpsp_sol.get_end_time(x) + rcpsp_model.index_task[x],
    )
    max_time = rcpsp_sol.get_end_time(sorted_task_by_end[-1])
    min_time = rcpsp_sol.get_start_time(sorted_task_by_start[0])
    patches = []
    pos = 0
    for j in range(nb_task):
        # nb_colors = len(tasks) // 2
        nb_colors = rcpsp_model.mode_details[len(tasks)][1]['org_task_id'] // 2
        colors = plt.cm.get_cmap("hsv", nb_colors)
        pos = rcpsp_model.mode_details[j + 1][1]['org_task_id'] - 1

        box = [
            (pos - 0.25, rcpsp_sol.rcpsp_schedule[tasks[j]]["start_time"]),
            (pos - 0.25, rcpsp_sol.rcpsp_schedule[tasks[j]]["end_time"]),
            (pos + 0.25, rcpsp_sol.rcpsp_schedule[tasks[j]]["end_time"]),
            (pos + 0.25, rcpsp_sol.rcpsp_schedule[tasks[j]]["start_time"]),
            (pos - 0.25, rcpsp_sol.rcpsp_schedule[tasks[j]]["start_time"]),
        ]
        polygon = Polygon([(b[1], b[0]) for b in box])
        x, y = polygon.exterior.xy
        ax.plot(x, y, zorder=-1, color=colors((pos) % nb_colors))
        patches.append(
            pp(xy=polygon.exterior.coords, facecolor=colors((pos) % nb_colors))
        )

    p = PatchCollection(
        patches,
        match_original=True,
        alpha=0.4,
    )
    ax.add_collection(p)
    if x_lim is None:
        ax.set_xlim((min_time, max_time))
    else:
        ax.set_xlim(x_lim)
    ax.set_ylim((-0.5, pos))
    ax.set_yticks(range(pos))
    ax.set_yticklabels(
        tuple([str(tasks[j]) for j in range(pos)]), fontdict={"size": 5}
    )
    ax.set_ylabel("Task number")
    ax.set_xlabel("Sprint")

    # print the sucessors with arrows
    pos = 0
    for j in range(nb_task):
        nb_colors = rcpsp_model.mode_details[len(tasks)][1]['org_task_id'] // 2
        colors = plt.cm.get_cmap("hsv", nb_colors)
        org_pos_y = rcpsp_model.mode_details[j + 1][1]['org_task_id'] - 1
        org_x_end_time = rcpsp_sol.rcpsp_schedule[tasks[j]]["end_time"]
        sucessors = rcpsp_model.successors[j + 1]
        facecolor = colors((org_pos_y) % nb_colors)
        for sucessor in sucessors:
            dst_y = rcpsp_model.mode_details[sucessor][1]['org_task_id'] - 1
            dst_x_start_time = rcpsp_sol.rcpsp_schedule[sucessor]["start_time"]
            con_style(ax, facecolor,
                      org_x_end_time, org_pos_y,
                      dst_x_start_time, dst_y,
                      "angle,angleA=-90,angleB=180,rad=0")

    return fig


def compute_schedule_per_resource_individual(
    rcpsp_model: RCPSPModel,
    rcpsp_sol: RCPSPSolution,
    resource_types_to_consider: List[str] = None,
):
    modes = rcpsp_model.build_mode_dict(rcpsp_sol.rcpsp_modes)
    if resource_types_to_consider is None:
        resources = rcpsp_model.resources_list
    else:
        resources = resource_types_to_consider
    sorted_task_by_start = sorted(
        rcpsp_sol.rcpsp_schedule,
        key=lambda x: 100000 * rcpsp_sol.get_start_time(x) + rcpsp_model.index_task[x],
    )
    sorted_task_by_end = sorted(
        rcpsp_sol.rcpsp_schedule,
        key=lambda x: 100000 * rcpsp_sol.get_end_time(x) + rcpsp_model.index_task[x],
    )
    max_time = rcpsp_sol.get_end_time(sorted_task_by_end[-1])
    min_time = rcpsp_sol.get_start_time(sorted_task_by_end[0])
    with_calendar = rcpsp_model.is_varying_resource()

    array_ressource_usage = {
        resources[i]: {
            "activity": np.zeros(
                (
                    max_time - min_time + 1,
                    rcpsp_model.get_max_resource_capacity(resources[i]),
                )
            ),
            "binary_activity": np.zeros(
                (
                    max_time - min_time + 1,
                    rcpsp_model.get_max_resource_capacity(resources[i]),
                )
            ),
            "total_activity": np.zeros(
                rcpsp_model.get_max_resource_capacity(resources[i])
            ),
            "activity_last_n_hours": np.zeros(
                (
                    max_time - min_time + 1,
                    rcpsp_model.get_max_resource_capacity(resources[i]),
                )
            ),
            "boxes_time": [],
        }
        for i in range(len(resources))
    }
    total_time = max_time - min_time + 1
    nhour = int(min(8, total_time / 2 - 1))
    index_to_time = {i: min_time + i for i in range(max_time - min_time + 1)}
    time_to_index = {index_to_time[i]: i for i in index_to_time}

    for activity in sorted_task_by_start:
        mode = modes[activity]
        start_time = rcpsp_sol.rcpsp_schedule[activity]["start_time"]
        end_time = rcpsp_sol.rcpsp_schedule[activity]["end_time"]
        if end_time == start_time:
            continue
        resources_needed = {
            r: rcpsp_model.mode_details[activity][mode].get(r, 0) for r in resources
        }
        for r in resources_needed:
            if r not in array_ressource_usage:
                continue
            rneeded = resources_needed[r]
            if not with_calendar:
                range_interest = range(array_ressource_usage[r]["activity"].shape[1])
            else:
                range_interest = range(
                    rcpsp_model.resources[r][time_to_index[start_time]]
                )
            while rneeded > 0:
                availables_people_r = [
                    i
                    for i in range_interest
                    if array_ressource_usage[r]["activity"][
                        time_to_index[start_time], i
                    ]
                    == 0
                ]
                logger.debug(f"{len(availables_people_r)} people available : ")
                if len(availables_people_r) > 0:
                    resource = min(
                        availables_people_r,
                        key=lambda x: array_ressource_usage[r]["total_activity"][x],
                    )
                    # greedy choice,
                    # the one who worked the less until now.
                    array_ressource_usage[r]["activity"][
                        time_to_index[start_time]: time_to_index[end_time], resource
                    ] = activity
                    array_ressource_usage[r]["binary_activity"][
                        time_to_index[start_time]: time_to_index[end_time], resource
                    ] = 1
                    array_ressource_usage[r]["total_activity"][resource] += (
                        end_time - start_time
                    )
                    array_ressource_usage[r]["activity_last_n_hours"][
                        :, resource
                    ] = np.convolve(
                        array_ressource_usage[r]["binary_activity"][:, resource],
                        np.array([1] * nhour + [0] + [0] * nhour),
                        mode="same",
                    )
                    array_ressource_usage[r]["boxes_time"] += [
                        [
                            (resource - 0.25, start_time + 0.01, activity),
                            (resource - 0.25, end_time - 0.01, activity),
                            (resource + 0.25, end_time - 0.01, activity),
                            (resource + 0.25, start_time + 0.01, activity),
                            (resource - 0.25, start_time + 0.01, activity),
                        ]
                    ]
                    # for plot purposes.
                    rneeded -= 1
                else:
                    logger.debug(f"r_needed {rneeded}")
                    logger.debug(f"Ressource needed : {resources_needed}")
                    logger.debug(f"ressource : {r}")
                    logger.debug(f"activity : {activity}")
                    logger.warning("Problem, can't build schedule")
                    logger.debug(array_ressource_usage[r]["activity"])
                    rneeded = 0

    return array_ressource_usage


def plot_resource_individual_gantt(
    rcpsp_model: RCPSPModel,
    rcpsp_sol: RCPSPSolution,
    resource_types_to_consider: List[str] = None,
    title_figure="",
    x_lim=None,
    fig=None,
    ax=None,
    current_t=None,
):
    array_ressource_usage = compute_schedule_per_resource_individual(
        rcpsp_model, rcpsp_sol, resource_types_to_consider=resource_types_to_consider
    )
    sorted_task_by_end = sorted(
        rcpsp_sol.rcpsp_schedule,
        key=lambda x: 100000 * rcpsp_sol.get_end_time(x) + rcpsp_model.index_task[x],
    )
    max_time = rcpsp_sol.get_end_time(sorted_task_by_end[-1])
    min_time = rcpsp_sol.get_start_time(sorted_task_by_end[0])
    resources_list = list(array_ressource_usage.keys())
    if fig is None or ax is None:
        fig, ax = plt.subplots(len(array_ressource_usage), figsize=(20, 10))
        fig.suptitle(title_figure)
        if len(array_ressource_usage) == 1:
            ax = [ax]

    tasks = rcpsp_model.tasks_list
    nb_colors = rcpsp_model.mode_details[len(tasks)][1]['org_task_id'] // 2
    colors = plt.cm.get_cmap("hsv", nb_colors)
    for i in range(len(resources_list)):
        patches = []
        for boxe in array_ressource_usage[resources_list[i]]["boxes_time"]:
            polygon = Polygon([(b[1], b[0]) for b in boxe])
            activity = boxe[0][2]
            org_pos_y = rcpsp_model.mode_details[activity][1]['org_task_id'] - 1
            color = colors((org_pos_y) % nb_colors)
            x, y = polygon.exterior.xy
            ax[i].plot(x, y, zorder=-1, color=color)
            patches.append(
                pp(
                    xy=polygon.exterior.coords,
                    facecolor=color,
                )
            )
            # Add box names
            box_name = str(rcpsp_model.mode_details[activity][1]['org_task_id'])
            box_center_x = (boxe[0][1] + boxe[-1][1]) / 2 + 0.5
            box_center_y = (boxe[0][0] + boxe[-1][0]) / 2 + 0.25
            ax[i].annotate(
                box_name,
                xy=(box_center_x, box_center_y),
                xytext=(box_center_x, box_center_y),
                fontsize=7,
                ha="center",
                va="center",
            )

        p = PatchCollection(
            patches,
            match_original=True,
            alpha=0.4,
        )
        ax[i].add_collection(p)
        ax[i].set_title(resources_list[i])
        if x_lim is None:
            ax[i].set_xlim((min_time, max_time))
        else:
            ax[i].set_xlim(x_lim)
        try:
            ax[i].set_ylim((-0.5, rcpsp_model.resources[resources_list[i]]))
            ax[i].set_yticks(range(rcpsp_model.resources[resources_list[i]]))
            ax[i].set_yticklabels(
                tuple([j for j in range(rcpsp_model.resources[resources_list[i]])]),
                fontdict={"size": 7},
            )
        except Exception as e:
            print(e)
            m = max(rcpsp_model.resources[resources_list[i]])
            ax[i].set_ylim((-0.5, m))
            ax[i].set_yticks(range(m))
            ax[i].set_yticklabels(tuple([j for j in range(m)]), fontdict={"size": 7})

        ax[i].grid(True)
        if current_t is not None:
            ax[i].axvline(x=current_t, label="pyplot vertical line", color="r", ls="--")
    ax[-1].set_xlabel("Sprint")
    return fig


def kendall_tau_similarity(rcpsp_sols: Tuple[RCPSPSolution, RCPSPSolution]):
    sol1 = rcpsp_sols[0]
    sol2 = rcpsp_sols[1]

    perm1 = sol1.generate_permutation_from_schedule()
    perm2 = sol2.generate_permutation_from_schedule()

    ktd, p_value = scipy.stats.kendalltau(perm1, perm2)
    return ktd


def intersect(i1, i2):
    if i2[0] >= i1[1] or i1[0] >= i2[1]:
        return None
    else:
        s = max(i1[0], i2[0])
        e = min(i1[1], i2[1])
        return [s, e]


def all_diff_start_time(rcpsp_sols: Tuple[RCPSPSolution, RCPSPSolution]):
    sol1 = rcpsp_sols[0]
    sol2 = rcpsp_sols[1]
    diffs = {}
    for act_id in sol1.rcpsp_schedule.keys():
        diff = (
            sol1.rcpsp_schedule[act_id]["start_time"]
            - sol2.rcpsp_schedule[act_id]["start_time"]
        )
        diffs[act_id] = diff
    return diffs


def compute_graph_rcpsp(rcpsp_model: RCPSPModel):
    nodes = [
        (
            n,
            {
                mode: rcpsp_model.mode_details[n][mode]["duration"]
                for mode in rcpsp_model.mode_details[n]
            },
        )
        for n in rcpsp_model.tasks_list
    ]
    edges = []
    for n in rcpsp_model.successors:
        for succ in rcpsp_model.successors[n]:
            dict_transition = {
                mode: rcpsp_model.mode_details[n][mode]["duration"]
                for mode in rcpsp_model.mode_details[n]
            }
            min_duration = min(dict_transition.values())
            max_duration = max(dict_transition.values())
            dict_transition["min_duration"] = min_duration
            dict_transition["max_duration"] = max_duration
            dict_transition["minus_min_duration"] = -min_duration
            dict_transition["minus_max_duration"] = -max_duration
            dict_transition["link"] = 1
            edges += [(n, succ, dict_transition)]
    return Graph(nodes, edges, False)


def save_task_gantt_resolution(
    rcpsp_model: RCPSPModel,
    rcpsp_sol: RCPSPSolution,
):

    rcpsp_gantt_solution = []
    tasks = rcpsp_model.tasks_list
    list_resources = rcpsp_model.resources_list

    c_past, times = compute_resource_consumption(
        rcpsp_model, rcpsp_sol, list_resources=list_resources, future_view=False
    )

    nb_task = len(tasks)
    for j in range(nb_task):
        rcpsp_model.mode_details[j + 1][1]['task_id'] = j + 1
        start_time = rcpsp_sol.rcpsp_schedule[tasks[j]]["start_time"]
        rcpsp_model.mode_details[j + 1][1]['start_time'] = start_time
        end_time = rcpsp_sol.rcpsp_schedule[tasks[j]]["end_time"]
        rcpsp_model.mode_details[j + 1][1]['end_time'] = end_time
        rcpsp_model.mode_details[j + 1][1]['Sucessors_task_id'] = rcpsp_model.successors[j + 1].copy()
        rcpsp_gantt_solution.insert(j, rcpsp_model.mode_details[j + 1][1].copy())

    rcpsp_gantt_solution_df = pd.DataFrame(rcpsp_gantt_solution)
    rcpsp_gantt_solution_df.to_csv('solution.csv', sep='\t')
    return c_past


def plot_task_gantt_plotly(
    rcpsp_model: RCPSPModel,
    rcpsp_sol: RCPSPSolution,
    title=None,
    current_t=None,
):
    tasks = rcpsp_model.tasks_list
    nb_task = len(tasks)
    sorted_task_by_start = sorted(
        rcpsp_sol.rcpsp_schedule,
        key=lambda x: 100000 * rcpsp_sol.get_start_time(x) + rcpsp_model.index_task[x],
    )
    sorted_task_by_end = sorted(
        rcpsp_sol.rcpsp_schedule,
        key=lambda x: 100000 * rcpsp_sol.get_end_time(x) + rcpsp_model.index_task[x],
    )

    max_time = rcpsp_sol.get_end_time(sorted_task_by_end[-1])
    min_time = rcpsp_sol.get_start_time(sorted_task_by_start[0])

    fig = make_subplots(rows=1, cols=1, vertical_spacing=0.05)
    fig.update_layout(title_text=title if title else "Gantt Task")

    shapes = []
    nb_colors = len(css_colors)
    
    for j in range(nb_task):
        pos = rcpsp_model.mode_details[j + 1][1]['org_task_id'] - 1
        color_idx = pos % nb_colors
        color = css_colors[color_idx]

        x = [
            rcpsp_sol.rcpsp_schedule[tasks[j]]["start_time"],
            rcpsp_sol.rcpsp_schedule[tasks[j]]["end_time"],
            rcpsp_sol.rcpsp_schedule[tasks[j]]["end_time"],
            rcpsp_sol.rcpsp_schedule[tasks[j]]["start_time"],
            rcpsp_sol.rcpsp_schedule[tasks[j]]["start_time"],
        ]
        y = [pos - 0.25, pos - 0.25, pos + 0.25, pos + 0.25, pos - 0.25]

        fig.add_trace(go.Scatter(x=x, y=y, mode='lines', fill='toself',
                                 fillcolor=color,
                                 line=dict(color=color), showlegend=False))

    if current_t:
        fig.add_shape(type="line", x0=current_t, y0=-0.5, x1=current_t, y1=pos, line=dict(dash='dash'))

    for j in range(nb_task):
        org_pos_y = rcpsp_model.mode_details[j + 1][1]['org_task_id'] - 1
        org_x_end_time = rcpsp_sol.rcpsp_schedule[tasks[j]]["end_time"]
        successors = rcpsp_model.successors[j + 1]
        pos = rcpsp_model.mode_details[j + 1][1]['org_task_id'] - 1
        for successor in successors:
            dst_y = rcpsp_model.mode_details[successor][1]['org_task_id'] - 1
            dst_x_start_time = rcpsp_sol.rcpsp_schedule[successor]["start_time"]
            shapes.append(
                dict(
                    type="path",
                    path=f"M {org_x_end_time},{org_pos_y} L {dst_x_start_time},{dst_y}",
                    line=dict(color='red'),
                    marker=dict(symbol='arrow', size=10)
                )
            )

    fig.update_layout(shapes=shapes)
    fig.update_xaxes(title_text="Sprint")
    fig.update_yaxes(title_text="Task number", ticktext=[str(tasks[j]) for j in range(pos)], tickfont=dict(size=5))

    return fig

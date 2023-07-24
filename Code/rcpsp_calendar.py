# cspell:disable
import os
from typing import Optional
from discrete_optimization.datasets import get_data_home
from discrete_optimization.rcpsp.rcpsp_model import (
    MultiModeRCPSPModel,
    RCPSPModel,
    RCPSPModelCalendar,
)
import copy


def get_data_available(
    data_folder: Optional[str] = None, data_home: Optional[str] = None
):
    """Get datasets available for rcpsp.

    Params:
        data_folder: folder where datasets for rcpsp whould be find.
            If None, we look in "rcpsp" subdirectory of `data_home`.
        data_home: root directory for all datasets. Is None, set by
            default to "~/discrete_optimization_data "

    """
    if data_folder is None:
        data_home = get_data_home(data_home=data_home)
        data_folder = f"{data_home}/rcpsp"

    try:
        files = [
            f for f in os.listdir(data_folder) if f.endswith(".sm") or f.endswith(".mm")
        ]
    except FileNotFoundError:
        files = []
    return [os.path.abspath(os.path.join(data_folder, f)) for f in files]


def find_task_position(mode_details_reordered, org_task_id):
    for task_id, details in mode_details_reordered.items():
        if details[1]['org_task_id'] == org_task_id:
            return task_id
    return None


def reorder_tasks(mode_details, successors):
    mode_details_reordered = {}
    successors_reordered = {}
    new_unit_task = 0

    for mode, details in mode_details.items():
        duration = details[1]['duration']

        if duration == 0:
            new_unit_task += 1
            mode_details_reordered[new_unit_task] = copy.deepcopy(details)

        for step in range(duration):
            new_unit_task += 1
            mode_details_reordered[new_unit_task] = copy.deepcopy(details)
            mode_details_reordered[new_unit_task][1]['org_task_id_array_index'] = step
            mode_details_reordered[new_unit_task][1]['duration'] = 1

    new_unit_task = 0
    for mode, details in mode_details.items():
        duration = details[1]['duration']

        if duration == 0:
            new_unit_task += 1
            successors_reordered[new_unit_task] = copy.deepcopy(successors[mode])
            for index, task_id_old in enumerate(successors_reordered[new_unit_task]):
                successors_reordered[new_unit_task][index] = find_task_position(mode_details_reordered, task_id_old)

        for step in range(duration):
            new_unit_task += 1
            successors_reordered[new_unit_task] = copy.deepcopy(successors[mode])
            for index, task_id_old in enumerate(successors_reordered[new_unit_task]):
                successors_reordered[new_unit_task][index] = find_task_position(mode_details_reordered, task_id_old)
            if step < duration - 1:
                successors_reordered[new_unit_task] = [new_unit_task + 1]

    return mode_details_reordered, successors_reordered


def parse_psplib_calendar(input_data):
    # parse the input
    lines = input_data.split("\n")

    # Retrieving section bounds
    horizon_ref_line_index = lines.index("RESOURCES") - 1

    prec_ref_line_index = lines.index("PRECEDENCE RELATIONS:")
    prec_start_line_index = prec_ref_line_index + 2
    duration_ref_line_index = lines.index("REQUESTS/DURATIONS:")
    prec_end_line_index = duration_ref_line_index - 2
    duration_start_line_index = duration_ref_line_index + 3
    res_ref_line_index = lines.index("RESOURCEAVAILABILITIES:")
    duration_end_line_index = res_ref_line_index - 2
    res_start_line_index = res_ref_line_index + 1

    # Parsing horizon
    tmp = lines[horizon_ref_line_index].split()
    horizon = int(tmp[2])

    # Parsing resource information
    tmp1 = lines[res_start_line_index].split()
    tmp2 = lines[res_start_line_index + 1].split()

    resources_tmp = {
        str(tmp1[(i * 2)]) + str(tmp1[(i * 2) + 1]): [int(tmp2[i])]
        for i in range(len(tmp2))
    }
    non_renewable_resources = [
        name for name in list(resources_tmp.keys()) if name.startswith("N")
    ]
    n_resources = len(resources_tmp.keys())

    # Parsing calendar
    # Dict[str, List[int]]
    for i in range(res_start_line_index + 2, len(lines) - 2):
        tmp_calendar = lines[i].split()
        for index, key in enumerate(resources_tmp.keys()):
            resources_tmp[key].append(int(tmp_calendar[index]))

    resources = {
        str(resource): resources_tmp[resource]
        for resource in resources_tmp
    }

    # Parsing precedence relationship
    multi_mode = False
    successors = {}
    for i in range(prec_start_line_index, prec_end_line_index + 1):
        tmp = lines[i].split()
        task_id = int(tmp[0])
        n_successors = int(tmp[2])
        successors[task_id] = [int(x) for x in tmp[3: (3 + n_successors)]]

    # Parsing mode and duration information
    mode_details = {}
    for i_line in range(duration_start_line_index, duration_end_line_index + 1):
        tmp = lines[i_line].split()
        if len(tmp) == 3 + n_resources:
            task_id = int(tmp[0])
            mode_id = int(tmp[1])
            duration = int(tmp[2])
            resources_usage = [int(x) if duration > 0 else 0 for x in tmp[3: (3 + n_resources)]]
        else:
            multi_mode = True
            mode_id = int(tmp[0])
            duration = int(tmp[1])
            resources_usage = [(int(x)) if duration > 0 else 0 for x in tmp[3: (3 + n_resources)]]
        if int(task_id) not in list(mode_details.keys()):
            mode_details[int(task_id)] = {}

        mode_details[int(task_id)][mode_id] = {}  # Dict[int, Dict[str, int]]
        mode_details[int(task_id)][mode_id]["duration"] = duration
        mode_details[int(task_id)][mode_id]["org_task_id"] = int(task_id)
        mode_details[int(task_id)][mode_id]["org_task_id_array_index"] = int(0)
        for i in range(n_resources):
            mode_details[int(task_id)][mode_id][list(resources.keys())[i]] = resources_usage[i]

    mode_details, successors = reorder_tasks(mode_details, successors)

    if multi_mode:
        problem = MultiModeRCPSPModel(
            resources=resources,
            non_renewable_resources=non_renewable_resources,
            mode_details=mode_details,
            successors=successors,
            horizon=horizon,
            horizon_multiplier=30,
        )

    else:
        problem = RCPSPModelCalendar(
            resources=resources,
            non_renewable_resources=non_renewable_resources,
            mode_details=mode_details,
            successors=successors,
            horizon=horizon,
            horizon_multiplier=30,
        )
        # problem.calendar_details

    return problem


def parse_file(file_path) -> RCPSPModel:
    with open(file_path, "r", encoding="utf-8") as input_data_file:
        input_data = input_data_file.read()
        rcpsp_model = parse_psplib_calendar(input_data)
        return rcpsp_model


def add_calendar_psplib(file_path):
    with open(file_path, "r", encoding="utf-8") as input_data_file:
        input_data = input_data_file.read()
        # parse the input
        lines = input_data.split("\n")

        # Retrieving section bounds
        horizon_ref_line_index = lines.index("RESOURCES") - 1
        res_ref_line_index = lines.index("RESOURCEAVAILABILITIES:")
        res_start_line_index = res_ref_line_index + 1
        # Parsing horizon
        tmp = lines[horizon_ref_line_index].split()
        horizon = int(tmp[2])

        resorce_line = res_start_line_index + 1

        for j in range(horizon - 1):
            lines.insert(resorce_line + j, lines[resorce_line])
        # at least one elment has to be diferent
        r1_limitation = int(lines[resorce_line].split()[0])
        r1_limitation_new = r1_limitation + 1
        resorce_line = resorce_line + horizon - 1
        lines[resorce_line] = lines[resorce_line].replace(str(r1_limitation), str(r1_limitation_new))
        # write the calendar
        new_file_path = file_path.replace(".sm", "_calendar.sm")
        new_file_path = new_file_path.replace(".mm", "_calendar.mm")
        with open(new_file_path, "w+", encoding="utf-8") as output_file:
            output_file.write("\n".join(lines))


if __name__ == "__main__":
    import glob
    folder_path = r".\Data\rcpsp"
    files_list = glob.glob(folder_path + "/*.sm") + glob.glob(folder_path + "/*.mm")
    # add calendar to the psplib data
    for files_path in files_list:
        add_calendar_psplib(files_path)

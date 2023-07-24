# cspell:disable
from __future__ import annotations

from typing import Union

from discrete_optimization.rcpsp.rcpsp_model import (MultiModeRCPSPModel,
                                                     RCPSPModelCalendar)
from skdecide.hub.domain.rcpsp.rcpsp_sk import MRCPSPCalendar, RCPSPCalendar

from Code.rcpsp_calendar import parse_file


def load_domain(file_path) -> MRCPSPCalendar:

    rcpsp_model: Union[MultiModeRCPSPModel, RCPSPModelCalendar] = parse_file(
        file_path
    )
    if isinstance(rcpsp_model, RCPSPModelCalendar):
        my_domain = RCPSPCalendar(
            resource_names=rcpsp_model.resources_list,
            task_ids=sorted(rcpsp_model.mode_details.keys()),
            tasks_mode=rcpsp_model.mode_details,
            successors=rcpsp_model.successors,
            max_horizon=rcpsp_model.horizon,
            resource_availability=rcpsp_model.resources,
            resource_renewable={
                r: r not in rcpsp_model.non_renewable_resources
                for r in rcpsp_model.resources_list
            },
        )
    elif isinstance(rcpsp_model, MultiModeRCPSPModel):
        my_domain = MRCPSPCalendar(
            resource_names=rcpsp_model.resources_list,
            task_ids=sorted(rcpsp_model.mode_details.keys()),
            tasks_mode=rcpsp_model.mode_details,
            successors=rcpsp_model.successors,
            max_horizon=rcpsp_model.horizon,
            resource_availability=rcpsp_model.resources,
            resource_renewable={
                r: r not in rcpsp_model.non_renewable_resources
                for r in rcpsp_model.resources_list
            },
        )

    return my_domain

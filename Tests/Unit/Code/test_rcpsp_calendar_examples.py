# cspell:disable
import unittest
from unittest.mock import MagicMock, patch

from Code.rcpsp_calendar_examples import do_singlemode_calendar


class TestGetDataAvailable(unittest.TestCase):
    @patch("Code.rcpsp_calendar_examples.load_domain")
    @patch("Code.rcpsp_calendar_examples.DOSolver")
    @patch("Code.rcpsp_calendar_examples.rollout_episode")
    @patch("Code.rcpsp_calendar_examples.from_last_state_to_solution")
    @patch("Code.rcpsp_calendar_examples.plot_task_gantt")
    @patch("Code.rcpsp_calendar_examples.plot_ressource_view")
    @patch("Code.rcpsp_calendar_examples.plot_resource_individual_gantt")
    @patch("Code.rcpsp_calendar_examples.plt.show")
    @patch("Code.rcpsp_calendar_examples.get_complete_path")
    def test_do_singlemode_calendar(self, mock_get_complete_path, mock_show, mock_individual_gantt, mock_ressource_view,
                                    mock_task_gantt, mock_from_last_state_to_solution, mock_rollout_episode,
                                    mock_dosolver, mock_load_domain):
        mock_domain = MagicMock()
        mock_load_domain.return_value = mock_domain
        mock_get_complete_path.return_value = 'j301_1_calendar.sm'
        mock_solver = MagicMock()
        mock_dosolver.return_value = mock_solver
        mock_states = [MagicMock(), MagicMock()]
        mock_actions = [MagicMock(), MagicMock()]
        mock_values = [MagicMock(), MagicMock()]
        mock_rollout_episode.return_value = (mock_states, mock_actions, mock_values)
        mock_solution = MagicMock()
        mock_from_last_state_to_solution.return_value = mock_solution

        do_singlemode_calendar()

        mock_load_domain.assert_called_once_with('j301_1_calendar.sm')
        mock_domain.set_inplace_environment.assert_called_once_with(False)
        mock_dosolver.assert_called_once()
        mock_solver.solve.assert_called_once()
        mock_rollout_episode.assert_called_once_with(
            domain=mock_domain,
            solver=mock_solver,
            from_memory=mock_domain.get_initial_state(),
            max_steps=500,
            outcome_formatter=unittest.mock.ANY
        )
        mock_from_last_state_to_solution.assert_called_once_with(mock_states[-1], mock_domain)
        mock_task_gantt.assert_called_once_with(mock_solution.problem, mock_solution)
        mock_ressource_view.assert_called_once_with(mock_solution.problem, mock_solution)
        mock_individual_gantt.assert_called_once_with(mock_solution.problem, mock_solution)
        mock_show.assert_called_once()

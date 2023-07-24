import os
import unittest
from unittest.mock import MagicMock, Mock, call, patch

from discrete_optimization.datasets import get_data_home
from discrete_optimization.rcpsp.rcpsp_model import RCPSPModelCalendar

from Code.rcpsp_calendar import (find_task_position, get_data_available,
                                 parse_psplib_calendar, reorder_tasks)


class TestGetDataAvailable(unittest.TestCase):

    def setUp(self):
        self.data_home = get_data_home()
        self.data_folder = f"{self.data_home}/rcpsp"

    def test_get_data_available_no_data_available(self):
        # Test with no data available
        with patch("Code.rcpsp_calendar.os.listdir", MagicMock(return_value=[])):
            data_available = get_data_available(data_folder=self.data_folder)
            self.assertEqual(data_available, [])

    @patch("Code.rcpsp_calendar.os.listdir", return_value=["file1.sm", "file2.mm", "other_file.txt"])
    @patch("Code.rcpsp_calendar.os.path.abspath", side_effect=lambda x: f"abspath_{x}")
    def test_get_data_available(self, abspath_mock: Mock, listdir_mock: Mock):
        expected = ["abspath_rcpsp\\file1.sm", "abspath_rcpsp\\file2.mm"]
        result = get_data_available(data_folder="rcpsp")
        self.assertEqual(result, expected)
        listdir_mock.assert_called_once_with("rcpsp")
        abspath_mock.assert_has_calls([call("rcpsp\\file1.sm"), call("rcpsp\\file2.mm")])


class TestFindtaskPosition(unittest.TestCase):
    def test_find_task_position(self):
        mode_details_reordered = {
            1: ('Task 1', {'org_task_id': 100}),
            2: ('Task 2', {'org_task_id': 200}),
            3: ('Task 3', {'org_task_id': 300})
        }
        org_task_id = 200
        expected_task_id = 2
        self.assertEqual(find_task_position(mode_details_reordered, org_task_id), expected_task_id)

    def test_find_task_position_no_match(self):
        mode_details_reordered = {
            1: ('Task 1', {'org_task_id': 100}),
            2: ('Task 2', {'org_task_id': 200}),
            3: ('Task 3', {'org_task_id': 300})
        }
        org_task_id = 400
        expected_task_id = None
        self.assertEqual(find_task_position(mode_details_reordered, org_task_id), expected_task_id)

    def test_find_task_position_empty_dict(self):
        mode_details_reordered = {}
        org_task_id = 200
        expected_task_id = None
        self.assertEqual(find_task_position(mode_details_reordered, org_task_id), expected_task_id)

    def test_find_task_position_duplicate_org_task_id(self):
        mode_details_reordered = {
            1: ('Task 1', {'org_task_id': 100}),
            2: ('Task 2', {'org_task_id': 200}),
            3: ('Task 3', {'org_task_id': 200})
        }
        org_task_id = 200
        expected_task_id = 2
        self.assertEqual(find_task_position(mode_details_reordered, org_task_id), expected_task_id)


class TestTaskReordering(unittest.TestCase):
    def setUp(self):
        self.mode_details = {
            1: ['mode1', {'org_task_id': 1, 'duration': 2, 'org_task_id_array_index': 0}],
            2: ['mode2', {'org_task_id': 2, 'duration': 0, 'org_task_id_array_index': 0}],
            3: ['mode3', {'org_task_id': 3, 'duration': 1, 'org_task_id_array_index': 0}],
            4: ['mode4', {'org_task_id': 4, 'duration': 3, 'org_task_id_array_index': 0}]
        }
        self.successors = {
            1: [2, 3],
            2: [4],
            3: [4],
            4: []
        }

    def test_reorder_tasks(self):
        expected_mode_details_reordered = {
            1: ['mode1', {'org_task_id': 1, 'duration': 1, 'org_task_id_array_index': 0}],
            2: ['mode1', {'org_task_id': 1, 'duration': 1, 'org_task_id_array_index': 1}],
            3: ['mode2', {'org_task_id': 2, 'duration': 0, 'org_task_id_array_index': 0}],
            4: ['mode3', {'org_task_id': 3, 'duration': 1, 'org_task_id_array_index': 0}],
            5: ['mode4', {'org_task_id': 4, 'duration': 1, 'org_task_id_array_index': 0}],
            6: ['mode4', {'org_task_id': 4, 'duration': 1, 'org_task_id_array_index': 1}],
            7: ['mode4', {'org_task_id': 4, 'duration': 1, 'org_task_id_array_index': 2}]
        }
        expected_successors_reordered = {
            1: [2],
            2: [3, 4],
            3: [5],
            4: [5],
            5: [6],
            6: [7],
            7: []
        }
        mode_details_reordered, successors_reordered = reorder_tasks(self.mode_details, self.successors)
        self.assertEqual(mode_details_reordered, expected_mode_details_reordered)
        self.assertEqual(successors_reordered, expected_successors_reordered)


class TestParsePsplibCalendar(unittest.TestCase):
    def setUp(self) -> None:
        current_file_path = os.path.abspath(__file__)
        current_directory = os.path.dirname(current_file_path)
        fixture_path = current_directory + '/Fixtures/j301_1_calendar.sm'
        with open(fixture_path, "r", encoding="utf-8") as input_data_file:
            self.input_data = input_data_file.read()

    def test_parse_psplib_calendar(self) -> None:
        problem = parse_psplib_calendar(self.input_data)
        assert isinstance(problem, RCPSPModelCalendar)
        assert problem.horizon == 6
        assert problem.resources == {'R1': [4, 4, 4, 10, 10, 5], 'R2': [13, 13, 0,
                                     13, 13, 13], 'R3': [4, 4, 4, 4, 4, 4],
                                     'R4': [12, 12, 12, 12, 12, 12]}

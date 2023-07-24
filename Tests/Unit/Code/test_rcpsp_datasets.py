from unittest import TestCase
from unittest.mock import patch

from Code.rcpsp_datasets import get_complete_path, get_data_available


class TestGetCompletePath(TestCase):
    @patch("Code.rcpsp_datasets.get_data_available")
    def test_get_complete_path(self, mock_get_data_available):
        # Mock the return value of get_data_available()
        mock_get_data_available.return_value = [
            "/path/to/rcpsp/j30.sm",
            "/path/to/rcpsp/j60.sm",
            "/path/to/rcpsp/j120.sm",
        ]

        # Test a valid input
        result = get_complete_path("j30")
        self.assertEqual(result, "/path/to/rcpsp/j30.sm")

        # Test an invalid input
        result = get_complete_path("foo")
        self.assertIsNone(result)


class TestDataAvailable(TestCase):

    def test_get_data_available(self):
        with patch("Code.rcpsp_datasets.os.listdir") as mock_listdir:
            mock_listdir.return_value = ["file1.txt", "file2.txt", "file3.csv", "file4.json", "file5.pk"]
            with patch("Code.rcpsp_datasets.path_to_data", "/path/to/data"):
                expected_output = ["/path/to/data\\file1.txt", "/path/to/data\\file2.txt", "/path/to/data\\file3.csv"]
                self.assertEqual(get_data_available(), expected_output)

    def test_get_data_available_with_no_files(self):
        with patch("Code.rcpsp_datasets.os.listdir") as mock_listdir:
            mock_listdir.return_value = []
            with patch("Code.rcpsp_datasets.path_to_data", "/path/to/data"):
                self.assertEqual(get_data_available(), [])

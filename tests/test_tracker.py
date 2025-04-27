from unittest.mock import patch, MagicMock
from trackers import Tracker

# Mocking YOLO initialization in the Tracker class
@patch('trackers.tracker.YOLO')
def test_tracker_init(mock_yolo):
    # Mock the return value of the YOLO object
    mock_yolo.return_value = MagicMock()
    
    # Initialize Tracker with a mock model path
    tracker = Tracker("models/best.pt")
    
    # Test that the model has been initialized without errors
    assert tracker.model is not None
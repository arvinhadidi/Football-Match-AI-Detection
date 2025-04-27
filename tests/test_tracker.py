from trackers import Tracker

def test_tracker_init():
    tracker = Tracker("models/best.pt")
    assert tracker is not None

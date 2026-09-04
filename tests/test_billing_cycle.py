from datetime import datetime, timezone
from app.billing_cycle import cycle_start


def test_the_cycle_starts_on_the_anchor_day_clamped_to_the_month():
    now = datetime(2026, 9, 3, 12, tzinfo=timezone.utc)
    assert cycle_start(31, now) == datetime(2026, 8, 31, tzinfo=timezone.utc)   # Brave's: Aug 31, "3/30 days"
    assert cycle_start(1, now) == datetime(2026, 9, 1, tzinfo=timezone.utc)
    assert cycle_start(31, datetime(2026, 10, 1, tzinfo=timezone.utc)) == datetime(2026, 9, 30, tzinfo=timezone.utc)  # clamped
    assert cycle_start(15, datetime(2026, 9, 14, tzinfo=timezone.utc)) == datetime(2026, 8, 15, tzinfo=timezone.utc)

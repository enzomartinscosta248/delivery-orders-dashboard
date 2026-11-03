import pandas as pd

from delivery_dashboard.insights import delay_summary, peak_summary


def test_peak_and_delay_summaries_describe_the_numbers():
    by_hour = pd.DataFrame(
        {
            "hour": [11, 12, 13, 19, 20, 21],
            "orders": [50, 120, 90, 110, 150, 100],
            "late_pct": [3.0, 9.0, 5.0, 14.0, 35.0, 16.0],
        }
    )
    heat = pd.DataFrame([[10, 30], [5, 80]], index=["Sex", "Sáb"], columns=[19, 20])

    peaks = peak_summary(by_hour, heat)
    assert "20h" in peaks[0] and "Sáb às 20h" in peaks[1] and "80" in peaks[1]

    by_neighborhood = pd.DataFrame({"neighborhood": ["Beira Rio", "Centro"], "late_pct": [18.0, 9.0]})
    delays = delay_summary(by_hour, by_neighborhood)
    assert "20h" in delays[0] and "35%" in delays[0]
    assert "sobrecarga" in delays[1]
    assert "Beira Rio" in delays[2] and "Centro" in delays[2]

    assert peak_summary(by_hour.iloc[0:0], heat) == []
    assert delay_summary(by_hour.iloc[0:0], by_neighborhood.iloc[0:0]) == []

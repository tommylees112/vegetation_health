import pandas as pd
from datetime import datetime

from predictor import CSVCleaner


def test_year_month():

    # lets predict June given a year's data
    months_2018 = [datetime(2018, x, 1) for x in range(1, 13)]
    months_2019 = [datetime(2019, x, 1) for x in range(1, 13)]

    g1, g2, g3 = [1] * 5, [2] * 12, [3] * 7
    test_data = {
        'times': months_2018 + months_2019,
        'group': g1 + g2 + g3,
    }
    test_df = pd.DataFrame(data=test_data)

    cleaner = CSVCleaner()
    test_df['gp_month'], test_df['gp_year'] = cleaner.update_year_month(test_df['times'],
                                                                        pred_month=5)

    # all of g2 should be in the same gp_year
    group_2 = test_df[test_df['gp_year'] == 2019]

    assert len(group_2) == 12, "Chopped out some 2018 data"
    assert (group_2['group'] == 2).all(), "Not all the correct months were grouped!"

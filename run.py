import fire
from pathlib import Path

from predictor import CSVCleaner, Engineer
from predictor.models import LinearModel, nn_FeedForward


class RunTask:

    @staticmethod
    def clean(raw_csv='data/raw/tabular_data.csv',
              processed_csv='data/processed/cleaned_data.csv',
              normalizing_percentile=95, pred_month=6):

        raw_csv, processed_csv = Path(raw_csv), Path(processed_csv)
        cleaner = CSVCleaner(raw_csv, processed_csv)
        cleaner.process(normalizing_percentile, pred_month)

    @staticmethod
    def engineer(cleaned_data='data/processed/cleaned_data.csv',
                 arrays='data/processed/arrays', test_year=2016):

        cleaned_data, arrays = Path(cleaned_data), Path(arrays)

        engineer = Engineer(cleaned_data, arrays)
        engineer.process(test_year)

    @staticmethod
    def train_model(model_type='feedforward', arrays='data/processed/arrays',
                    hide_vegetation=True):

        arrays = Path(arrays)

        string2model = {
            'baseline': LinearModel(arrays, hide_vegetation),
            'feedforward': nn_FeedForward(arrays, hide_vegetation)
        }

        model = string2model[model_type]
        model.train()
        model.evaluate()


if __name__ == '__main__':
    fire.Fire(RunTask)

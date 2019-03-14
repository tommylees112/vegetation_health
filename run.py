import fire
from pathlib import Path

from predictor import CSVCleaner, Engineer
from predictor.models import LinearModel


class RunTask:

    @staticmethod
    def clean(raw_csv='data/raw/tabular_data.csv',
              processed_csv='data/processed/cleaned_data.csv',
              normalizing_percentile=95):

        raw_csv, processed_csv = Path(raw_csv), Path(processed_csv)
        cleaner = CSVCleaner(raw_csv, processed_csv)
        cleaner.process(normalizing_percentile)

    @staticmethod
    def engineer(cleaned_data='data/processed/cleaned_data.csv',
                 arrays='data/processed/arrays'):

        cleaned_data, arrays = Path(cleaned_data), Path(arrays)

        engineer = Engineer(cleaned_data, arrays)
        engineer.process()

    @staticmethod
    def train_model(model_type='baseline', arrays='data/processed/arrays'):

        arrays = Path(arrays)

        string2model = {
            'baseline': LinearModel(arrays),
        }

        model = string2model[model_type]
        model.train()


if __name__ == '__main__':
    fire.Fire(RunTask)

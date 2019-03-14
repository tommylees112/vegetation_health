import fire

from predictor import CSVCleaner


class RunTask:

    @staticmethod
    def clean(raw_csv='data/raw/tabular_data.csv',
              processed_csv='data/processed/cleaned_data.csv',
              normalizing_percentile=95):

        cleaner = CSVCleaner(raw_csv, processed_csv)
        cleaner.process(normalizing_percentile)


if __name__ == '__main__':
    fire.Fire(RunTask)

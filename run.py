import fire
from pathlib import Path

from predictor import CSVCleaner, NCCleaner, Engineer
from predictor.models import LinearModel, nn_FeedForward, nn_Recurrent


class RunTask:

    @staticmethod
    def clean(raw_filepath='data/raw/tabular_data.csv',
              processed_filepath='data/processed/cleaned_data.csv',
              target='ndvi_anomaly', pred_month=6, netcdf=False, ):

        raw_filepath, processed_filepath = Path(raw_filepath), Path(processed_filepath)
        if netcdf:
            cleaner = NCCleaner(raw_filepath, processed_filepath)
        else:
            cleaner = CSVCleaner(raw_filepath, processed_filepath)
            
        cleaner.process(pred_month, target)

    @staticmethod
    def engineer(cleaned_data='data/processed/cleaned_data.csv',
                 arrays='data/processed/arrays', test_year=2016):

        cleaned_data, arrays = Path(cleaned_data), Path(arrays)

        engineer = Engineer(cleaned_data, arrays)
        engineer.process(test_year)

    @staticmethod
    def train_model(model_type='baseline', arrays='data/processed/arrays',
                    hide_vegetation=True):

        arrays = Path(arrays)

        string2model = {
            'baseline': LinearModel(arrays, hide_vegetation),
            'feedforward': nn_FeedForward(arrays, hide_vegetation),
            'recurrent': nn_Recurrent(arrays, hide_vegetation),
        }

        model = string2model[model_type]
        model.train()
        model.evaluate()


if __name__ == '__main__':
    fire.Fire(RunTask)

import fire
from pathlib import Path

from predictor import Cleaner, Engineer
from predictor.models import LinearModel, nn_FeedForward, nn_Recurrent


class RunTask:

    @staticmethod
    def clean(raw_filepath='data/raw/predict_vegetation_health.nc',
              processed_folder='data/processed',
              target='ndvi', pred_month=6):

        raw_filepath, processed_folder = Path(raw_filepath), Path(processed_folder)
        processed_filepath = processed_folder / target / 'cleaned_data.csv'

        cleaner = Cleaner(raw_filepath, processed_filepath)
        cleaner.process(pred_month, target)

    @staticmethod
    def engineer(processed_folder='data/processed', target='ndvi',
                 test_year=2016):

        processed_folder = Path(processed_folder)
        cleaned_data = processed_folder / target / 'cleaned_data.csv'
        arrays_folder = processed_folder / target / 'arrays'

        engineer = Engineer(cleaned_data, arrays_folder)
        engineer.process(test_year)

    @staticmethod
    def train_model(model_type='baseline', data_folder='data',
                    target='ndvi', hide_vegetation=True, save_results=True):

        data_folder = Path(data_folder)
        arrays_folder = data_folder / 'processed' / target / 'arrays'

        string2model = {
            'baseline': LinearModel(data_folder, arrays_folder, hide_vegetation),
            'feedforward': nn_FeedForward(data_folder, arrays_folder, hide_vegetation),
            'recurrent': nn_Recurrent(data_folder, arrays_folder, hide_vegetation),
        }

        model = string2model[model_type]
        model.train()
        model.evaluate(save_preds=save_results)
        if save_results:
            model.save_model()


if __name__ == '__main__':
    fire.Fire(RunTask)

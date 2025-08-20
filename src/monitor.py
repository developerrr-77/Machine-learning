from comet_ml import Experiment

def log_experiment(product, mae, rmse, mape):
    experiment = Experiment(project_name="inventory-predictions")
    experiment.log_metrics({'mae': mae, 'rmse': rmse, 'mape': mape})
    experiment.end()
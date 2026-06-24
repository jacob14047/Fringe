import mlflow
from datetime import datetime

class MLflowLogger:
    def __init__(self, experiment_name="bb84_experiments", tracking_uri="sqlite:///mlflow.db"):
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
    
    def start_run(self, run_name=None):
        if run_name is None:
            run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        return mlflow.start_run(run_name=run_name)
    
    def log_params(self, params):
        mlflow.log_params(params)
    
    def log_metrics(self, metrics, step=None):
        mlflow.log_metrics(metrics, step=step)
    
    def log_artifact(self, local_path):
        mlflow.log_artifact(local_path)
    
    def end_run(self):
        mlflow.end_run()
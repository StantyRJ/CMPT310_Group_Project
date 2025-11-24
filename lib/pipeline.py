import numpy as np
from tqdm import tqdm

def run_training(model, dataset_provider, eval_fn=None):
    """Generic pipeline: dataset and model are injected.

    Train the model with the provided dataset, then determine it's accuracy.

    - model: must implement fit(X_train,y_train) and predict(X_test)
    - dataset_provider: must implement load() -> X_train,y_train,X_test,y_test
    - eval_fn: optional function(preds, y_true) -> metrics dict
    """
    X_train, y_train, X_test, y_test = dataset_provider.load()

    tqdm.write(f"Loaded dataset — train {len(X_train)} samples, test {len(X_test)} samples")

    tqdm.write("Fitting model...")
    model.fit(X_train, y_train)
    tqdm.write("Predicting...")
    preds = model.predict(X_test)

    preds = np.asarray(preds)
    y_test = np.asarray(y_test)
    acc = float((preds == y_test).mean())
    tqdm.write(f"Accuracy: {acc:.4f}")

    metrics = {"accuracy": acc}
    if eval_fn is not None:
        metrics.update(eval_fn(preds, y_test))
    return metrics

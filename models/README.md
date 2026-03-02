# Models Directory

This directory stores trained model artifacts.

## Model Files

1. **xgboost_model.pkl** - XGBoost classifier (stored on Google Drive)
2. **distilbert/** - Fine-tuned DistilBERT model directory (stored on Google Drive/HuggingFace)

## Note

Model files are NOT committed to git due to their size. They are generated during training:

```bash
python main.py train-models
```

Models will be saved to this directory after training.

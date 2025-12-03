# Amazon Food Review Sentiment Analysis

# Data Directory

## Structure

```
data/
├── raw/                # Original dataset from Kaggle
└── processed/          # Cleaned & preprocessed data for training
```

## Getting the Dataset

### Amazon Fine Food Reviews Dataset

Download the dataset from Kaggle:

**URL:** https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews

**Steps:**

1. Create a Kaggle account at https://www.kaggle.com
2. Download the dataset (Reviews.csv)
3. Place the CSV file in the `data/raw/` directory
4. Run preprocessing script: `python src/preprocessing.py`

## Dataset Info

- **File:** Reviews.csv (~1.5 GB)
- **Records:** 568,454 reviews
- **Columns:** ProductId, UserId, ProfileName, HelpfulnessNumerator, HelpfulnessDenominator, Score, Time, Summary, Text
- **Target Variable:** Score (1-5 rating) → Converted to Negative/Neutral/Positive

## Preprocessing Output

After running `src/preprocessing.py`, the `processed/` folder will contain:

- `train_data.csv` - Training set (cleaned reviews)
- `test_data.csv` - Test set (cleaned reviews)
- `val_data.csv` - Validation set (cleaned reviews)

## Note

Do NOT commit the raw data files to GitHub. The `.gitignore` file ensures large data files are not tracked.

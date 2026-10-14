# econ3916-lab06-sampling

# Sampling for Machine Learning — Train-Test Splits

## Objective
I used the Titanic data and a simulated 1936 election to show how the way a sample is drawn affects a model's results more than how big the sample is.

## Methodology
- Loaded the Titanic dataset (891 passengers, `Survived` as the target) and dropped the 177 rows with a missing Age, leaving 714 passengers with an overall survival rate of 0.406.
- Compared survival rates by passenger class (63.0% in 1st class, 47.3% in 2nd, 24.2% in 3rd) to see why class matters for splitting.
- Ran a naive 80/20 train-test split and a stratified one (`stratify=y`), then compared survival rates and passenger class shares in each half.
- Stratified on a combined Pclass + Survived label so that both the class mix and the survival rate carry over to the train and test sets.
- Reproduced data leakage with `StandardScaler`, fitting it on all rows before the split, then fixed it by splitting first and fitting on the training rows only.
- Did the same for `SimpleImputer` on Age: fit the mean on the full data (29.6991), then on the training rows only (29.8077), and applied that training value to the test rows.
- Wrote `split_by_share(data, test_share, seed)`, a plain split with no stratification. I used it for a 30% split of the cleaned data and checked it against the lab's own 80/20 split.
- Simulated 10 million voters for the 1936 Literary Digest poll, with a biased sample (chance of being sampled rises with income) and a simple random sample of 50,000.
- Computed Meng's data defect correlation and used it to predict the biased sample's error.
- Repeated the biased vs. random comparison 100 times with a new seed each time.

## Key Findings
- **Stratifying kept the survival rate steady.** On the naive split the test set's survival rate was 0.392 against 0.406 overall. With stratification, both halves came out at 0.406. Stratifying on Survived alone did not fix the class mix (the naive test set was 53.8% 3rd class vs. 49.7% overall), but stratifying on Pclass and Survived together matched the class shares to within 0.003.
- **Leakage was small here but real.** The scaler fit on all rows learned an Age mean of 29.6991, compared with 29.3325 when fit on training rows only. For Fare the gap was about 1.0. Leaving this in would make test accuracy look better than it should, because the test rows influenced the preprocessing.
- **`split_by_share()` worked.** It passed the lab's check. The 30% split gave 499 training rows (survival rate 0.403) and 215 test rows (0.414).
- **Sample design beat sample size.** The biased Literary Digest sample had 4,382,503 responses and predicted a Landon share of 0.636, an error of +0.083. The random Gallup-style sample of 50,000 predicted 0.550, an error of −0.002 (true share: 0.553). The biased sample was about 90 times larger and still less accurate.
- **Meng's formula explained the error.** A data defect correlation of 0.1475 produced an error of 8.3 percentage points, and the formula's prediction matched the actual error to four decimals.
- **The defect correlation here is far larger than in real surveys.** I set the sampling mechanism to be strongly income-driven, so the real-world effect would be smaller, but the mechanism is the same.

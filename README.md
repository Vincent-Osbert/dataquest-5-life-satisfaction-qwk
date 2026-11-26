# DataQuest 5.0 — Life Satisfaction Prediction (LifeScope)

Predict a person's life satisfaction on a 4-level scale (1 = very satisfied, 4 = very dissatisfied) from a national health survey. Built for DataQuest 5.0, Universitas Airlangga, as team **Warkab**. The team placed **4th in the national final**.

| | |
|---|---|
| Out-of-fold QWK (final pipeline) | **0.48425** |
| Gain from expected-value thresholds over argmax | +0.061 QWK |
| Train / test rows | 95,609 / 23,902 |
| Features | 176 numeric survey answers |

## Ringkasan untuk HR dan pembaca non-IT

**Masalahnya.** Dari jawaban survei kesehatan nasional (95.609 orang, 176 pertanyaan), tebak seberapa puas seseorang dengan hidupnya dalam 4 tingkat: sangat puas, puas, tidak puas, sangat tidak puas.

**Yang saya kerjakan.** Bersama tim bertiga, membangun beberapa model dan menggabungkannya. Temuan terpenting: cara memutuskan "tingkat mana" lebih menentukan daripada model mana. Mengganti aturan keputusan saja menaikkan skor 0,06, sedangkan menambah model hanya menambah 0,004. Kurva belajar juga menunjukkan batasnya ada di pertanyaan survei, bukan di model.

**Hasilnya.** Peringkat 4 di final nasional DataQuest 5.0 (Universitas Airlangga). Skor QWK 0,484, naik 17% dari satu model LightGBM biasa (0,414).

**Gunanya di perusahaan.** Penilaian bertingkat seperti ini dipakai untuk survei kepuasan pelanggan atau karyawan, skor risiko nasabah (rendah sampai sangat tinggi), dan menentukan siapa yang perlu dihubungi lebih dulu. Proyek ini juga menunjukkan kapan menambah data lebih berguna daripada mengganti model, sehingga waktu tim tidak terbuang.

**Kemampuan yang terbukti.** Klasifikasi bertingkat (ordinal) · ensemble model pohon · optimasi ambang keputusan · membaca batas data dari kurva belajar.

## The problem

The metric is quadratic weighted kappa (QWK), which punishes a prediction by the square of its distance from the true class. The target is badly imbalanced: 44.9% class 1, 50.4% class 2, 3.7% class 3 and 0.9% class 4. A model that predicts class 2 for everyone gets 50.4% accuracy and QWK 0.

## Approach

**EDA decisions** (notebook sections 3.1–3.10)
- 14.41% of cells are missing, and the missing pattern in train and test is almost identical, so missing values are left as `NaN` for the tree models instead of being imputed.
- 84 yes/no columns contain the survey codes 7, 8 and 9 (refused, not ascertained, don't know). They are kept as values. Turning them into `NaN` changed 3-fold QWK by only +0.0011.
- Members of the same household appear in both train and test, which looks like a target-encoding opportunity. Knowing that another member answered "very satisfied" raises the chance of the same answer only from 44.9% to about 46.5%, so household labels are not used.
- 13 composite features (counts of difficulties, diseases, cost barriers, mental-health severity) were tested in an ablation and dropped: they changed QWK by −0.0009.

**Modelling**
1. A vectorised QWK (`qwk_fast`) matches scikit-learn to 4.4e-16 and runs 48 times faster, which makes threshold search cheap.
2. Six base models share one 5-fold stratified split, so their out-of-fold (OOF) predictions can be stacked: LightGBM, CatBoost and XGBoost classifiers, a LightGBM Huber regressor, an ordinal model built from three cumulative LightGBM classifiers (y > 1, y > 2, y > 3) and an MLP.
3. Predictions are turned into an expected value E[y], which is cut with three thresholds instead of taking the most likely class.
4. The final combiner is the plain average of the five tree models, fixed in advance rather than picked by OOF score.
5. The thresholds are the average of bootstrap estimates, also fixed in advance. In a split-half test this strategy lost the least between in-sample and held-out QWK.

## Results

All numbers are out-of-fold on the 95,609 training rows.

| Model | QWK | Accuracy |
|---|---|---|
| Majority class | 0.00000 | 0.504 |
| Logistic regression + thresholds | 0.44880 | 0.600 |
| Single LightGBM + argmax | 0.41448 | 0.667 |
| Single LightGBM + thresholds | 0.47717 | 0.635 |
| **Final pipeline** (tree average + bootstrap thresholds) | **0.48425** | 0.633 |

The decision rule mattered more than the model. On the same tree average, argmax gives QWK 0.42460 and expected value with thresholds gives 0.48517, a gain of 0.06057. Accuracy drops at the same time, from 0.669 to 0.633, because the thresholds push more predictions into the rare classes 3 and 4, where a correct guess is worth the most under a squared penalty. Stacking the models added only 0.004 over the best single model.

The five tree models land between 0.47847 and 0.48114, a spread of 0.003. A learning curve explains why:

| Training rows | 11,473 | 23,902 | 43,024 | 66,926 | 95,609 |
|---|---|---|---|---|---|
| QWK | 0.43883 | 0.46162 | 0.46391 | 0.46911 | 0.47669 |

A log fit, QWK = 0.29209 + 0.01613·ln(n), explains the curve with R² 0.914. Each doubling of the data adds about 0.011 QWK. Reaching QWK 0.60 would take about 195 million rows, 2,041 times the current data. The limit is in what the survey asks, not in the model.

For the final presentation the team also built a browser demo with a compact surrogate model: 440 trees on 17 features, QWK 0.46752, which is 96.5% of the full pipeline. The demo app lives in a teammate's repository: [MikaelAdikara/DataQuest-Warkab](https://github.com/MikaelAdikara/DataQuest-Warkab) ([live demo](https://mikaeladikara.github.io/DataQuest-Warkab/)).

## Limitations

- **OOF is the only score here.** No test-set or leaderboard score is reported in this repository.
- **Duplicates.** Train contains 1,892 fully duplicated rows, and the stratified split does not keep duplicates in the same fold, so the OOF score may be slightly optimistic.
- **Rare classes stay hard.** Class 3 has recall 0.38 and precision 0.21; class 4 has recall 0.30 and precision 0.18.
- **The extrapolation is rough.** The data-size estimates come from a log fit on five points. The curve is flattening, so the real requirement is likely higher.
- **Outputs come from a rerun.** The notebook was re-run end to end for this repository on 4 November 2026 (Python 3.12.5, versions from `requirements.txt`, 146 minutes on CPU), and the outputs shown are from that run. The output that printed raw respondent rows was removed.

## Repository structure

```
notebooks/dataquest5-warkab-life-satisfaction.ipynb
requirements.txt        pinned versions plus technical notes
```

The notebook text is in Indonesian. The competition data, the trained model (84 MB) and the submission file are not included.

## How to run

Put `train_final.csv`, `test_final.csv` and `data_dictionary.csv` next to the notebook (or in `./data`), install `requirements.txt` and run all cells. The notebook writes the model and `submission.csv` to `final/`. A full run takes about 2 to 2.5 hours on CPU.

## Credits

Team Warkab: Mikael Alexander Adikara Purnama, Vincent Osbert and Rico Majesty Daniel Mitra. The competition and data belong to the DataQuest 5.0 organisers at Universitas Airlangga.

# Part B: Business Case Analysis
## Promotion Effectiveness at a Fashion Retail Chain

---

## B1. Problem Formulation

### B1(a) — ML Problem Formulation (3 marks)

**Target Variable:**
The target variable is `items_sold` — the number of items sold in a given store during a given month under a specific promotion.

**Candidate Input Features:**

| Feature Category | Examples |
|-----------------|---------|
| Store attributes | store_id, store_size, location_type (urban/semi-urban/rural), monthly_footfall |
| Promotion details | promotion_type (Flat Discount, BOGO, Free Gift, Category Offer, Loyalty Points) |
| Calendar features | month, is_weekend, is_festival, is_month_end |
| Competition | competition_density (local competition level) |
| Customer demographics | average_age_group, income_bracket (if available) |
| Historical performance | avg_items_sold_last_3_months, best_promotion_last_month |

**Type of ML Problem:**
This is a **supervised regression problem** — we are predicting a continuous numeric outcome (items_sold) based on labelled historical data.

However, it also has an **optimisation dimension**: once the regression model can predict items_sold for any (store, month, promotion) combination, we can recommend the promotion that yields the highest predicted items_sold for each store-month pair.

**Justification:** Since the target (items_sold) is a continuous count variable and we have historical labelled data, regression is the most appropriate framing. If only the best promotion were needed (without the predicted count), it could also be framed as multi-class classification — but regression provides richer information for business decisions.

---

### B1(b) — Why Items Sold is Better Than Revenue (3 marks)

**Why items_sold is more reliable:**

1. **Price variability distorts revenue:** Revenue = items_sold × price. If the company runs a Flat Discount, the same number of items sold generates lower revenue — making it look like the promotion underperformed when it actually drove equal or higher volume.

2. **BOGO promotions create double-counting:** A Buy-One-Get-One offer technically records two items sold, but only one was paid for. Revenue would misrepresent performance, while items_sold more directly captures customer response to the promotion.

3. **Promotions affect price, not just quantity:** Since promotions themselves change the price, using revenue as a target creates a confounded feedback loop. Items sold is independent of the price manipulation.

**Broader principle — Target Variable Selection:**
This illustrates the principle of **choosing a target variable that is causally clean and not manipulated by the input features**. In real-world ML, the target should measure what the business truly cares about and should not be influenced by the very inputs we are using to predict it. When the target and features are not causally independent, models learn spurious correlations rather than true patterns.

---

### B1(c) — Alternative to a Single Global Model (2 marks)

**Problem with one global model:**
Stores in urban, semi-urban, and rural locations attract very different customer profiles and respond differently to the same promotion. A single model trained on all 50 stores would average out these differences, producing recommendations that are mediocre for every store and optimal for none.

**Proposed Alternative — Stratified or Hierarchical Modelling:**

**Option 1 — Segmented Models (recommended for simplicity):**
Train a separate model for each location type (urban, semi-urban, rural). Each model learns the promotion-response dynamics specific to that segment. This is simple to implement and interpretable.

**Option 2 — Mixed Effects / Hierarchical Model:**
Train one model with store_id as a categorical feature and use interaction terms between `store_id` and `promotion_type`. This allows the model to learn store-specific promotion sensitivities while sharing information across stores (useful when individual stores have limited data).

**Option 3 — Store-Level Models (for large stores with sufficient history):**
For flagship stores with years of data, train individual store models. For smaller stores with less history, fall back to the segmented model.

**Justification:** A segmented model strikes the best balance between accuracy and practicality. It accounts for the structural differences between location types while keeping the number of models manageable (3 models instead of 50).

---

## B2. Data and EDA Strategy

### B2(a) — Table Joining and Grain (4 marks)

**How to join the four tables:**

The four source tables are:
- **Transactions** — one row per transaction (transaction_id, store_id, date, items_sold, promotion_id)
- **Store Attributes** — one row per store (store_id, store_size, location_type, footfall, competition_density)
- **Promotion Details** — one row per promotion (promotion_id, promotion_type, discount_percentage)
- **Calendar** — one row per date (date, is_weekend, is_festival, month, day_of_week)

**Join sequence:**
```
Transactions
  LEFT JOIN Store Attributes   ON transactions.store_id = store_attributes.store_id
  LEFT JOIN Promotion Details  ON transactions.promotion_id = promotion_details.promotion_id
  LEFT JOIN Calendar           ON transactions.date = calendar.date
```

**Grain of the final modelling dataset:**
> One row = one store × one month × one promotion type

**Aggregations performed before modelling:**
- Sum `items_sold` across all transactions for that store-month-promotion combination
- Average `competition_density` (stable, but take monthly average in case it varies)
- Flag `is_festival` as 1 if any day in that month had a festival (binary OR aggregation)
- Count `is_weekend` days in the month (sum of weekend days)
- Keep store attributes as they are (static per store)

This grain ensures each row represents a complete, comparable unit: what happened at a specific store in a specific month when a specific promotion was run.

---

### B2(b) — EDA Strategy (4 marks)

**1. Target Variable Distribution:**
- Plot a histogram and boxplot of `items_sold`.
- **What to look for:** Skewness, outliers, and whether a log transformation is needed. Extreme outliers (e.g., a festival month with 10× normal sales) could unduly influence regression — and may warrant separate treatment or a robust model.

**2. Promotion Type vs Items Sold (Box Plots):**
- Create side-by-side boxplots of `items_sold` grouped by `promotion_type`.
- **What to look for:** Which promotion consistently yields highest median items_sold? Large variance within a promotion type suggests the effect is moderated by other features (e.g., location), indicating interaction terms are needed.

**3. Heatmap — Promotion Performance by Location Type:**
- Create a pivot table: rows = promotion_type, columns = location_type, values = mean items_sold. Visualise as a heatmap.
- **What to look for:** Whether the best promotion differs across urban/rural/semi-urban locations. If it does, this confirms the need for segmented models (as argued in B1c).

**4. Temporal Trend Analysis:**
- Plot total items_sold over time (month-year on x-axis).
- **What to look for:** Seasonality (e.g., peaks in November-December for festivals), long-term growth or decline trends. Seasonal patterns inform feature engineering — we should include `month` and `is_festival` as features and may need seasonal decomposition.

**5. Correlation Analysis:**
- Compute a correlation matrix between numerical features and `items_sold`.
- **What to look for:** Strong correlates (e.g., `footfall`, `competition_density`) to prioritise as features. Near-zero correlates can be deprioritised. High inter-feature correlations signal multicollinearity issues for linear models.

**How findings influence modelling decisions:**
- Skewed target → apply log transformation before training
- Interaction effects found → add interaction features (e.g., promotion × location_type)
- Seasonality found → include month, is_festival, lag features
- Multicollinearity → use regularised regression (Ridge/Lasso) or tree-based models

---

### B2(c) — Handling Promotion Imbalance (2 marks)

**How the imbalance affects the model:**
If 80% of transactions have no promotion (`promotion_type = None`), the model is trained overwhelmingly on non-promotion behaviour. It will learn the baseline sales pattern well but may generalise poorly to promoted scenarios. The model may under-predict the lift from promotions, making all promotions appear less effective than they are.

**Steps to address it:**

1. **Separate analysis for promoted vs non-promoted records:** Build the model specifically on promoted transactions. The 80% non-promotion data can still be useful for baseline benchmarking but shouldn't dominate training if the goal is to rank promotions against each other.

2. **Use promotion-specific features carefully:** Ensure the model sees enough examples of each promotion type. If some promotions appear in very few rows, consider **oversampling** those promotion types (for classification) or adding a class weight — though for regression, SMOTE equivalents are less standard.

3. **Stratified splitting:** When creating train/test sets, stratify by `promotion_type` to ensure each promotion appears proportionally in both sets — preventing a situation where the test set contains almost no examples of a rare promotion.

4. **Separate baseline model:** Train a baseline model on no-promotion transactions to predict expected sales without any promotion. Then model the **promotion lift** (items_sold_with_promo − baseline_prediction) as the target — this isolates the promotion's contribution and avoids imbalance issues.

---

## B3. Model Evaluation and Deployment

### B3(a) — Train-Test Split and Metrics (4 marks)

**Setting up the train-test split:**
With 3 years of monthly data across 50 stores, we have approximately 36 months × 50 stores = 1,800 store-month rows. A time-based split is essential:

- **Training set:** First 27–30 months (75–80% of time period)
- **Test set:** Last 6–9 months (most recent period)
- **Validation approach:** Use **time-series cross-validation** (walk-forward validation) during training — train on months 1–12, validate on month 13; then train on months 1–13, validate on month 14, and so on. This prevents data leakage and mimics real deployment.

**Why random split is inappropriate:**
A random split would mix past and future observations. The model could train on October data and test on September data — effectively seeing "the future" during training. This leads to inflated performance metrics that do not reflect real-world predictive ability. In deployment, we always predict the future based on the past, so our evaluation must replicate that structure.

**Evaluation Metrics:**

| Metric | Formula | Business Interpretation |
|--------|---------|------------------------|
| **RMSE** (Root Mean Squared Error) | √(mean(actual − predicted)²) | Average prediction error in units of items_sold. Penalises large errors heavily — important when big mistakes (e.g., over-stocking) are costly. |
| **MAE** (Mean Absolute Error) | mean(|actual − predicted|) | Average absolute error. More intuitive — "on average, our prediction is off by X items." Less sensitive to outliers than RMSE. |
| **R²** (Coefficient of Determination) | 1 − (SS_res / SS_tot) | What proportion of variance in items_sold is explained by the model? R² = 0.85 means 85% of variation is captured. A value below 0.6 suggests the model needs improvement. |
| **MAPE** (Mean Absolute % Error) | mean(|actual − predicted| / actual) × 100 | Percentage error — useful for comparing performance across stores of very different sizes. A 10% MAPE means predictions are off by 10% on average. |

---

### B3(b) — Feature Importance for Explaining Recommendations (4 marks)

**Scenario:** The model recommends Loyalty Points Bonus for Store 12 in December, but Flat Discount for Store 12 in March.

**How to investigate using feature importance:**

1. **Global feature importance:** First, look at the Random Forest's overall feature importance scores. This tells us which features the model relies on most across all stores and months — e.g., `is_festival`, `month`, `competition_density`.

2. **Local explainability with SHAP values:** Use SHAP (SHapley Additive exPlanations) to explain individual predictions. For Store 12 in December, SHAP shows how much each feature pushed the prediction toward Loyalty Points Bonus. For March, it shows which features changed.

3. **Likely explanation:** December is a festival month (`is_festival = 1`) with high footfall — customers are already motivated to buy. Loyalty Points Bonus works better here because customers are making large purchases and want future rewards. In March (off-peak, `is_festival = 0`), footfall is lower and customers need an immediate incentive — hence Flat Discount works better by reducing the perceived barrier to purchase.

**Communicating to the marketing team:**

Present a simple table or visual like:

| Feature | December Value | March Value | Impact |
|---------|---------------|-------------|--------|
| is_festival | 1 (Yes) | 0 (No) | Large ↑ for Loyalty Bonus |
| month | 12 (December) | 3 (March) | Seasonal pattern |
| competition_density | Low | High | Drives need for instant discount |

Frame the explanation in business language: *"In December, customers are in a buying mindset due to festivals — they respond to loyalty rewards that keep them coming back. In March, footfall drops and competition is higher, so an immediate price reduction is needed to drive purchases."* Avoid technical jargon like "SHAP values" in communications with non-technical stakeholders.

---

### B3(c) — End-to-End Deployment Process (4 marks)

**Step 1 — Save the Model:**
After training, serialise the full scikit-learn Pipeline (including the preprocessor and model) using `joblib`:
```python
import joblib
joblib.dump(rf_pipeline, 'promotion_recommender_v1.pkl')
```
Saving the entire pipeline ensures preprocessing steps are applied consistently — no risk of forgetting to scale or encode new data. Store the model file in a versioned location (e.g., AWS S3, Google Cloud Storage, or a company model registry).

**Step 2 — Preparing New Monthly Data:**
At the start of every month:
1. Pull the latest store, transaction, and calendar data from the data warehouse
2. Aggregate to store-month grain (same aggregations as training — see B2a)
3. Create all engineered features (month, is_festival, is_month_end, etc.)
4. Load the saved pipeline and call `pipeline.predict(new_data)` — preprocessing is applied automatically
5. For each store, generate predictions for all 5 promotion types and recommend the one with the highest predicted items_sold

**Step 3 — Monitoring for Model Degradation:**
Deploy a monitoring dashboard that tracks the following signals monthly:

| Monitor | Method | Action Threshold |
|---------|--------|-----------------|
| **Prediction Accuracy** | Compare predictions to actual items_sold once month ends. Track RMSE and MAE. | If RMSE increases >20% above training baseline over 3 consecutive months → retrain |
| **Feature Drift** | Monitor the distribution of input features (e.g., mean footfall, competition_density) using statistical tests (KS test, PSI). | If PSI > 0.2 for a key feature → investigate and consider retraining |
| **Recommendation Distribution** | Track which promotion is recommended across stores. | If one promotion suddenly dominates (e.g., 90% of stores → Loyalty Points) when it previously was 30%, flag as anomalous model behaviour |
| **Business Override Rate** | Track how often the marketing team overrides the model's recommendation. | High override rate signals the model is out of sync with business reality |

**Retraining Strategy:**
- **Scheduled retraining:** Retrain every 6 months on a rolling 2-year window of data to keep the model current with shifting consumer behaviour.
- **Triggered retraining:** Retrain immediately if monitoring signals a statistically significant drop in accuracy.
- **A/B testing:** Before replacing the current model, run the new model in parallel for 1–2 months on a subset of stores to confirm improvement before full rollout.

🚀 Spaceship Titanic — Advanced ML Classification Pipeline

A machine learning system built for the Kaggle Spaceship Titanic competition, predicting whether passengers were transported to an alternate dimension during a spacetime anomaly.

This project focuses heavily on:

* leakage-free preprocessing,
* intelligent missing value imputation,
* domain-driven feature engineering,
* ensemble modeling,
* and robust cross-validation methodology.

The final stacking ensemble achieved 81.06% cross-validation accuracy with highly stable generalization across folds.

⸻

📌 Problem Statement

Given passenger records containing:

* demographics,
* cabin assignments,
* CryoSleep status,
* onboard spending behavior,
* and travel group information,

predict whether a passenger was Transported during the anomaly event.

The challenge involved:

* extensive missing values,
* noisy group relationships,
* sparse behavioral patterns,
* and preventing target leakage during preprocessing.

⸻

🧠 Key Ideas Behind the Solution

Instead of relying on generic preprocessing, the pipeline was designed around domain logic and relational structure inside the dataset.

Example:
Passengers in CryoSleep physically cannot spend money.

This insight was used to deterministically resolve large portions of missing data before any statistical imputation.

⸻

⚙️ Technical Pipeline

1. Intelligent Missing Value Imputation

Tier 1 — Domain Logic

* CryoSleep → spending columns filled with 0
* Any passenger with spending > 0 → inferred CryoSleep=False

This resolved most spending nulls without statistical estimation.

⸻

Tier 2 — Group-Based Imputation

Passengers sharing the same GroupID were analyzed using:

* last-name agreement,
* cabin sharing,
* age spread,
* home planet consistency.

A custom family confidence score determined whether values should propagate aggressively or conservatively across the group.

⸻

Tier 3 — Statistical Fallback

Remaining missing values filled using subgroup statistics:

* Age → median by HomePlanet × Destination
* categorical columns → subgroup mode

All statistics were fit on training data only to prevent leakage.

⸻

🛠️ Feature Engineering

Created 40+ engineered features across multiple categories.

Spending Features

* TotalSpend
* LuxurySpend
* SpendEntropy
* SpendConcentration
* ServicesUsed
* ZeroSpender

Relative Spend Features

* SpendVsGroup
* SpendVsPlanet
* SpendVsDestination

CryoSleep Interaction Features

* AgeCryo
* LuxuryCryo
* CryoViolation

Group & Family Features

* IsSolo
* IsGroupLeader
* FamilyScore
* GroupAgeStd
* CabinConflict

Age Features

* Age buckets
* IsChild
* IsElderly
* ChildSpending

Cabin Features

* Deck
* Side
* DeckSide
* Normalized cabin number

⸻

🔒 Leakage Prevention

A major focus of the project.

Initial CV scores reached nearly 1.0, which triggered a systematic leakage investigation.

The issue:

* Leave-One-Out encoding on GroupID
* Small groups caused near-perfect target memorization

Fixes applied:

* Removed high-cardinality LOO encoding
* Used target encoding only on low-cardinality categoricals
* Split data before all target-dependent transforms
* Added leakage diagnostics before training

⸻

🤖 Models

Base Models

* LightGBM
* XGBoost
* CatBoost

Hyperparameter Optimization

* Optuna Bayesian Optimization
* 100 tuning trials for LightGBM

Final Ensemble

A stacking classifier using:

* gradient boosting models as base learners
* logistic regression as the meta-learner

⸻

📊 Results
| Model | CV Accuracy | CV Std |
| --- | --- | --- |
| LightGBM | ~0.8068 |  |
| XGBoost | ~0.8079 |  |
| CatBoost | ~0.8051 |  |
| Stacking Ensemble | ~0.8106 | 0.0040 |

The extremely low variance across folds indicates strong generalization stability.

⸻

📈 Tech Stack

* Python
* Pandas
* NumPy
* Scikit-learn
* LightGBM
* XGBoost
* CatBoost
* Optuna
* SHAP
* Category Encoders
* SciPy

⸻

🧪 What I Learned

This project reinforced that:

* preprocessing quality matters more than model complexity on tabular data,
* leakage prevention is critical,
* domain logic can outperform naive statistical imputation,
* and suspiciously high CV scores should always be investigated.

The largest gains came not from deeper models, but from:

* better handling of missingness,
* structured feature engineering,
* and robust validation discipline.

⸻

🚀 Future Improvements

With more time, I would explore:

* FT-Transformer / TabNet models
* nested cross-validation for the full stacking pipeline
* pseudo-labeling on high-confidence predictions
* deeper error analysis on the remaining ~19% failures


# Predictive Modeling for Early Identification of Cardiovascular Heart Failure (CHF) Candidates

1. Objective
   
Develop a supervised machine learning model to predict the likelihood of CHF diagnosis up to three months before it occurs, enabling proactive care management and early non-surgical interventions.
________________________________________
2. Background

Heart stent surgeries represent a major cost driver in healthcare. Early identification of members at high risk for heart stent surgeries enables timely intervention through medication, and lifestyle management — all of which have been shown to reduce surgical necessity and improve patient outcomes.

However, current claims data show that only 50% of patients has CHF related medication within 12 months prior to surgery, indicating a gap in preventive management. Many patients remain untreated until their condition deteriorates, resulting in costly and avoidable surgical interventions.

By using predictive analytics, we aim to flag high-risk members for early outreach by care managers and physicians, facilitating conservative management and potential cost savings.
________________________________________
3. Machine Learning Pipeline
   
3.1 Data Sources and Feature Engineering

The model is trained on claims-based medical history at the member level.

Feature categories include:

	Demographics: Age, gender
	Financial metrics: Total medical expense, revenue, medical expense ratio
	Clinical features: Chronic conditions, ICD-10 diagnosis codes, DRG procedure codes, medication history, ER utilization
  
A total of 3,571 engineered features were extracted.

For ICD-10 codes, two derived feature sets were created:

	Frequency features: Number of occurrences of each diagnosis to represent condition severity
	Recency features: Days since last diagnosis, capturing the persistence of conditions
  
The target variable is binary:

	1 = Member underwent heart stent surgery
	0 = Member did not undergo heart stent surgery
  
The dataset is highly imbalanced, with a positive-to-negative ratio of approximately 0.15.
________________________________________
3.2 Model Development

Approach 1 – Tabular Machine Learning

Two time-based datasets were constructed to prevent temporal data leakage:
	Training set: Jan 2021 – Dec 2024 (~300k samples)
	Testing set: Jan 2025 – May 2025 (~80k samples)
  
Each record corresponds to a unique member with an event date (surgery date for surgery cases or a random reference date for non-surgical members). Claims features such as diagnosis, procedures, and pharmacy records were aggregated per member.

Approach 2 – NLP-Based Feature Representation

To address high feature dimensionality and potential overfitting, a Natural Language Processing (NLP) approach was employed.

Tabular claims data were converted into text-like sequences, and an N-gram TF-IDF representation (1–3 grams) was generated using TfidfVectorizer from scikit-learn.

Mathematical formulation:

	Term Frequency (TF): TF(t)=  (Number of occurances of term t)/(Total terms in the document)

  Inverse Document Frequency (IDF): IDF(t) = log (Total number of documents)/(Number of documents containing term t)
	
  TF-IDF: A weighted metric emphasizing terms that are frequent within a document but rare across the corpus, capturing both local and global term relevance.
________________________________________
4. Model Evaluation

While similar studies typically report AUROC as the main performance metric, our business use case prioritizes precision in identifying true positives (members likely to have surgery). Therefore, AUPRC (Area Under the Precision-Recall Curve) is used as the primary evaluation metric, as it better reflects performance on imbalanced datasets.
Precision (PPV) is also emphasized to quantify the probability that a flagged member truly undergoes surgery.

Catboost outperforms other models (AUPRC = 0.88, PPV = 0.72). 

In contrast, other models do not perform as well (xgboost: AUPRC = 0.76, PPV = 0.68; random forest: AUPRC = 0.75, PPV = 0.80; logistic regression: AUPRC = 0.54; PPV = 0.58) 		
________________________________________
5. Feature Importance and Model Interpretability

Key predictive features include indicators of hypertension, hyperlipidemia, obesity, and metabolic disorders—conditions.

Model interpretability was achieved using SHAP (SHapley Additive exPlanations) to provide member-level insights, enabling clinicians to understand which features drive individual predictions.
________________________________________

6. Results and Clinical Insights
	Best Model: CatBoost. Average Precision (AUPRC): 0.88. PPV: 0.72

	High-risk profile: Members presenting with obesity, frailty, chronic back pain, hypertension, and hyperlipidemia were significantly more likely to undergo heart stent surgery.

	Predictive accuracy:

      Between Jan–May 2025, the model correctly predicted 1,179 members who underwent surgery.
   
	    704 were identified as high risk, of which 357 (30%) had no record of CHF medication.
   
	    Early intervention for these 357 members could have potentially avoided surgery, yielding an estimated cost savings of $2.14M over 5 months.
   
  Additional analyses of ICD-10 frequencies and medication patterns are included to support clinical decision-making and care management planning.
________________________________________
7. Reproducibility and Notebooks
   
text_ngram.py: NLP pipeline functions (tokenization, TF-IDF generation)

chf_predictive_model.ipynb: 
	- Convert NLP features into structured tabular form
	- Model training and SHAP-based explainability
________________________________________
8. Conclusion
   
This predictive modeling framework demonstrates the potential of claims-based machine learning to identify members at elevated risk of HKR surgery.

By integrating interpretable models with clinical workflows, healthcare teams can target early interventions, reduce unnecessary surgeries, and generate measurable cost savings.

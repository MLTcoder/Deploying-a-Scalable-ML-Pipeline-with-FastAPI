# Model Card

## Model Details
•	Developed by Melissa Thomas, a student of WGU/Udacity for a Machine Learning DevOps project in November 2025.
•	Model is a Random Forest Classifier with 100 trees and entropy criteria.
## Intended Use
•	Model predicts if U.S. Census respondents earn more or less than 50K annually.
•	It is intended for educational purposes only, specifically to practice ML pipeline development and deployment using FastAPI.
•	It is not intended for hiring, salary, insurance, or housing decisions.
•	It is not intended for any judgement against individuals.
•	It is not intended for any sort of production or business system.
## Training Data
•	Data was trained on a 80% portion of a U.S. Census Adult dataset.
## Evaluation Data
•	Data was evaluated on a 20% portion of a U.S. Census Adult dataset.
•	No synthetic augmentation was used in the evaluation.
## Metrics
•	Precision, Recall, and the F1 Score were used as metrics to look for false positives and false negatives.
•	Precision overall was 0.7353.
•	Recall overall was 0.6378.
•	F1 overall was 0.6831.
•	Performance was also evaluated across all categorical feature values, with results contained in a separate file (slice_output.txt).
## Ethical Considerations
•	The data contains historical and demographic information which may contain, and enable, harmful biases against gender, race, and occupation. It should not be used in real-world decision-making.
•	The model reflects correlations, not necessarily causations. 
•	Again, this model is only to be used in academic learning.
## Caveats and Recommendations
•	The dataset will gradually become more outdated as time continues to pass. It would require updates for accurate future use.
•	Normalization and more advanced preprocessing steps were withheld from the model. With more time and attention, the model could be improved.
•	Future work could include applying fairness techniques, such as equalized odds, tuning hyperparameters with cross-validation, and evaluating it on additional, external, datasets.


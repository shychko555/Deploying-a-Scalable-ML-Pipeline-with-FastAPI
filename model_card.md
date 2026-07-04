# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
This model is a Random Forest Classifier trained using the scikit-learn library. It is configured with 100 trees to ensure robust classification performance on the provided census dataset.

## Intended Use
This model is intended for predicting whether an individual's annual income exceeds $50,000 based on census attributes. It is designed to assist in demographic analysis.

## Training Data
The model was trained on the Census Income dataset. The data was processed using an 80/20 train-test split, and categorical features were transformed using one-hot encoding.

## Evaluation Data
The model was evaluated using the held-out 20% test split from the original Census Income dataset.

## Metrics
The model performance was evaluated on the test set using the following metrics:
* **Precision**: 0.7455
* **Recall**: 0.6321
* **F1-score**: 0.6841

## Ethical Considerations
This model is trained on census data, which may contain historical biases related to socioeconomic status, occupation, or demographic groups. Users should be aware that predictions might reflect these societal biases rather than purely individual characteristics.

## Caveats and Recommendations
The model is intended to assist in salary prediction based on census data. It should not be used as the sole basis for high-stakes financial or hiring decisions without human oversight.


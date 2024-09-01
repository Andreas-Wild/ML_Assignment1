# Dry Bean Classification Project

This project focuses on classifying different types of dry beans using machine learning techniques. It includes data preprocessing, feature selection, and model training.

## Dependencies

This project requires the following Python libraries:

- pandas
- numpy
- matplotlib
- scikit-learn
- imbalanced-learn

You can install these dependencies using pip:
pip install pandas numpy matplotlib scikit-learn imbalanced-learn

## Detailed Dependencies

1. **Data Manipulation and Analysis:**
   - pandas
   - numpy

2. **Data Visualization:**
   - matplotlib.pyplot

3. **Machine Learning and Data Preprocessing:**
   - scikit-learn:
     - feature_selection (SelectKBest, f_classif)
     - ensemble (RandomForestClassifier)
     - preprocessing (LabelEncoder, OneHotEncoder, StandardScaler)
     - model_selection (train_test_split, cross_val_score)
     - neighbors (KNeighborsClassifier)
     - pipeline (Pipeline)
     - compose (ColumnTransformer)
     - impute (SimpleImputer)
     - metrics (confusion_matrix, classification_report, ConfusionMatrixDisplay)
     - decomposition (PCA)
     - tree (DecisionTreeClassifier)

4. **Handling Imbalanced Data:**
   - imbalanced-learn:
     - over_sampling (SMOTE)

## Usage

1. Ensure all dependencies are installed.
2. Open and run the Jupyter notebook `dry_bean_classification.ipynb` in the `src/` directory.
3. The notebook uses the dataset `DryBeanDataset.xlsx`, which should be in the same directory as the notebook.

## Note

Make sure you have Jupyter Notebook or JupyterLab installed to run the `.ipynb` file. If not, you can install it using:
pip install jupyter

Then, navigate to the `src/` directory and run:
jupyter notebook

This will open the Jupyter interface in your default web browser, where you can open and run the notebook.

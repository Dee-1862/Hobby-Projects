import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, jaccard_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix

def data_cleaning(data):
    if 'msno' in data.columns:
        # "msno" is dropped as it is a unique identifier and not a useful variable
        data.drop(["msno"], axis=1, inplace=True)

    # Encoding the 'gender' feature
    data['gender'] = data['gender'].map({'male': 1, 'female': 0})


    # Removes missing values in the 'gender' column
    imputer = SimpleImputer(missing_values=np.nan, strategy='median')
    data['gender'] = imputer.fit_transform(data[['gender']]).astype(int)

    return data



def date_cnvrsn(data):
    # Converts all the date columns to "%Y%m%d" (date only) format
    dt_columns = ['registration_init_time', 'transaction_date', 'membership_expire_date']
    for coln in dt_columns:
        if not pd.api.types.is_datetime64_any_dtype(data[coln]):
            data[coln] = pd.to_datetime(data[coln], format='%Y%m%d', errors='coerce')
    
    return data



def feature_engineering(data):
    # Total activity (Total songs played)
    data['total_activity'] = data[['num_25', 'num_50', 'num_75', 'num_985', 'num_100']].sum(axis=1)

    # Completion rate
    data['completion_rate'] = data['num_100'] / data['total_activity']

    # Creates a categorical column answering if the user is on a discounted plan
    data['is_discounted'] = (data['plan_list_price'] > data['actual_amount_paid']).astype(int)

    # Days since registration (Days from registration date to transaction date)
    data['regis_to_trans'] = (data['transaction_date'] - data['registration_init_time']).dt.days

    # Membership duration (Days from transaction date to membership expiry date)
    data['membership_duration'] = (data['membership_expire_date'] - data['transaction_date']).dt.days

    # Interaction ratio (Number of unique streams / Total songs played)
    data['interaction_ratio'] = data['num_unq'] / data['total_activity']

    # Extracts year and month from 'transaction_date'
    data['transaction_year'] = data['transaction_date'].dt.year
    data['transaction_month'] = data['transaction_date'].dt.month

    # Extracts year and month from 'registration_init_time'
    data['registration_year'] = data['registration_init_time'].dt.year
    data['registration_month'] = data['registration_init_time'].dt.month

    return data



def dropper(data):
    # Drops all the columns in the droppers list
    droppers = ['registration_init_time', 'transaction_date', 'membership_expire_date', 'date']
    data.drop(columns = droppers, inplace = True)

    return data



def histogram_plot(data, feature):
    # Group data and count instances for histogram
    hist_data = data.groupby([feature, 'is_churn']).size().reset_index(name='count')

    # creating a figure composed of two matplotlib.Axes objects (ax_box and ax_hist)
    fig, (ax_box, ax_hist) = plt.subplots(2, sharex = True, figsize=(11, 7), gridspec_kw={"height_ratios": (.15, .85)})

    # Boxplot
    sns.boxplot(
        data[feature],
        palette=["#1F77B5", "#E74C3E"],
        orient="h",
        ax = ax_box
    )
    ax_box.set(xlabel='')

    # Histogram plot
    sns.histplot(
        data = hist_data,
        x=feature,
        weights = 'count',
        hue='is_churn',
        multiple='stack',
        palette=["#1F77B5", "#E74C3E"],
        ax = ax_hist
    )
    plt.title(f'Distribution of {feature} over Churn')
    plt.show()



def lower_bound(data):
    data = data[data['bd'] > 0]
    num_var = ['num_25','num_50','num_75','num_985','num_100','num_unq','total_secs','total_activity','plan_list_price','actual_amount_paid','interaction_ratio','completion_rate','membership_duration','regis_to_trans']
    
    # Remove rows where any of the numerical variables have values < 0
    for x in num_var:
        if x in data.columns:
            data = data[data[x] >= 0]
    return data



def bd_outlier(data):
    Q1 = data['bd'].quantile(0.25)
    Q3 = data['bd'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    filtered_data = data[(data['bd'] >= lower_bound) & (data['bd'] <= upper_bound)]
    return filtered_data



def categorical_conversion (data):
    categorical_features = ['city','gender','registered_via','payment_method_id','is_auto_renew','is_cancel','is_discounted']

    # Converts categorical features to 'category' dtype
    for feature in categorical_features:
        if feature in data.columns:
            data[feature] = data[feature].astype('category')
    return data



def bar_plot(data, feature):
    bar_data = data.groupby([feature, 'is_churn']).size().reset_index(name='count')
    plt.figure(figsize=(11, 7))
    sns.barplot(data = bar_data, x = feature, y = 'count', hue = 'is_churn', palette = ["#1F77B5", "#E74C3E"])
    plt.title(f'Churn Rate by {feature}')
    plt.xlabel(feature)
    plt.ylabel('Number of Users')
    plt.show()



def transformation(data):
    num_features = ['bd', 'payment_plan_days', 'plan_list_price', 'actual_amount_paid',
                      'num_25', 'num_50', 'num_75', 'num_985', 'num_100', 'num_unq',
                      'total_secs', 'total_activity', 'completion_rate', 'interaction_ratio',
                      'membership_duration', 'regis_to_trans']
    scaler = MinMaxScaler()
    for x in num_features:
        data[x] = scaler.fit_transform(data[[x]])
    return data
        


def stratkfold(data):
    # Define features and target variable
    if 'is_churn' in data.columns:
        X_scaled = data.drop('is_churn', axis=1)
    y = data['is_churn'].cat.codes

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=69)

    # Perform Stratified K-Fold Cross-Validation
    for train_index, test_index in skf.split(X_scaled, y):
        X_train, X_test = X_scaled.iloc[train_index], X_scaled.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    return X_train, X_test, y_train, y_test, X_scaled



def plot_confusion_matrix(y_true, y_pred, labels, title, normalize=True, cmap="Blues"):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    # Plotting
    plt.figure(figsize=(11, 7))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap=cmap, xticklabels=labels, yticklabels=labels, cbar_kws={'label': 'Percentage (%)'})
    plt.title(title)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.show()



# Function for Model Evaluation
def model(tech, tech_name, X_train, X_test, y_train, y_test, parmtrs={}):
    model = tech(**parmtrs)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    def model_output(tech_name, y_test, y_pred):
        print(f"{tech_name}:")
        print(f"Accuracy: {accuracy_score(y_test, y_pred)}") # Correct predictions/Total predictions
        print(f"Precision: {precision_score(y_test, y_pred)}") # TP/(TP+FP)
        print(f"Recall: {recall_score(y_test, y_pred)}") # TP/(TP+FN)
        print(f"F1 Score: {f1_score(y_test, y_pred)}") # Gives the harmonic mean of precision and recall
        print(f"Jaccard Score: {jaccard_score(y_test, y_pred)}") # Measures the overlap between the true positive cases and the predicted positive cases
        plot_confusion_matrix(y_test, y_pred, labels=[0, 1], title=f"{tech_name} Confusion Matrix")
    model_output(tech_name, y_test, y_pred)
    return model
    


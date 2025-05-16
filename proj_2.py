from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 0. Import dataset
file_path = r'C:\Users\johan\OneDrive - IPSA\Bureau\A4\Intro to Machine Learning\Projet_crime_prediction\Dataset.csv'
crime_data = pd.read_csv(file_path)

# 1. Label encode target
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
crime_data['Primary_Type_Label'] = le.fit_transform(crime_data['Primary Type'])

# 2. Select features
# Convert 'Date' column to datetime
crime_data['Date'] = pd.to_datetime(crime_data['Date'], format='%m/%d/%Y %I:%M:%S %p', errors='coerce')

crime_data = crime_data.dropna(subset=['Date']) # Drop invalid rows 

crime_data['Year'] = crime_data['Date'].dt.year
crime_data['Month'] = crime_data['Date'].dt.month
crime_data['Day'] = crime_data['Date'].dt.day
crime_data['DayOfWeek'] = crime_data['Date'].dt.dayofweek
crime_data['Hour'] = crime_data['Date'].dt.hour
crime_data['IsWeekend'] = crime_data['DayOfWeek'].isin([5, 6]).astype(int)
crime_data['Date'] = pd.to_datetime(crime_data[['Year', 'Month', 'Day']])

crime_data['Hour_sin'] = np.sin(2 * np.pi * crime_data['Hour'] / 24) # Cyclic encoding to avoid discontinuity
crime_data['Hour_cos'] = np.cos(2 * np.pi * crime_data['Hour'] / 24)

crime_data['Month_sin'] = np.sin(2 * np.pi * crime_data['Month'] / 12)
crime_data['Month_cos'] = np.cos(2 * np.pi * crime_data['Month'] / 12)

crime_data['DayOfWeek_sin'] = np.sin(2 * np.pi * crime_data['DayOfWeek'] / 7)
crime_data['DayOfWeek_cos'] = np.cos(2 * np.pi * crime_data['DayOfWeek'] / 7)


# Day is an ordinal counter when DayOfWeek captures seasonality only
feature_cols = [
    'Year', 'Month_sin', 'Month_cos', 'Day', 'Hour_sin', 'Hour_cos', 'DayOfWeek_sin', 'DayOfWeek_cos', 'IsWeekend', 
    'District', 'Ward', 'Community Area', 
    'Latitude', 'Longitude',
    'Location Description', 'Domestic', 'Arrest'
]

crime_counts = crime_data['Primary Type'].value_counts().sort_values(ascending=False)
#print(crime_counts) # We remark that the dataset is not cleaned yet, some categories appear multiple times with different casing or spaces

# 2.1. Clean data
crime_data['Primary Type'] = (
    crime_data['Primary Type']
    .str.strip()                     # remove leading/trailing spaces
    .str.upper()                     # make consistent casing
    .str.replace(r'\s+', ' ', regex=True)  # collapse multiple spaces
)

crime_mapping = {
   
    'THEFT': 'PROPERTY CRIME',
    'BURGLARY': 'PROPERTY CRIME',
    'ROBBERY': 'PROPERTY CRIME',
    'MOTOR VEHICLE THEFT': 'PROPERTY CRIME',
    'CRIMINAL DAMAGE': 'PROPERTY CRIME',
    'ARSON': 'PROPERTY CRIME',
    'CRIMINAL TRESPASS': 'PROPERTY CRIME',

    'DECEPTIVE PRACTICE': 'FRAUD & ECONOMIC CRIME',
    'FORGERY': 'FRAUD & ECONOMIC CRIME',

    'BATTERY': 'VIOLENT CRIME',
    'ASSAULT': 'VIOLENT CRIME',
    'HOMICIDE': 'VIOLENT CRIME',
    'KIDNAPPING': 'VIOLENT CRIME',
    'STALKING': 'VIOLENT CRIME',
    'INTIMIDATION': 'VIOLENT CRIME',

    'NARCOTICS': 'DRUG OFFENSE',
    'OTHER NARCOTIC VIOLATION': 'DRUG OFFENSE',

    'DOMESTIC VIOLENCE': 'OFFENSE INVOLVING CHILDREN',

    'WEAPONS VIOLATION': 'WEAPONS OFFENSE',
    'CONCEALED CARRY LICENSE VIOLATION': 'WEAPONS OFFENSE',

    'PUBLIC PEACE VIOLATION': 'PUBLIC DISTURBANCE',
    'INTERFERENCE WITH PUBLIC OFFICER': 'PUBLIC DISTURBANCE',
    'GAMBLING': 'PUBLIC DISTURBANCE',
    'LIQUOR LAW VIOLATION': 'PUBLIC DISTURBANCE',
    'RITUALISM': 'PUBLIC DISTURBANCE',
    'OBSCENITY': 'PUBLIC DISTURBANCE',
    'PUBLIC INDECENCY': 'PUBLIC DISTURBANCE',

    'SEX OFFENSE': 'SEXUAL OFFENSE',
    'CRIM SEXUAL ASSAULT': 'SEXUAL OFFENSE',
    'CRIMINAL SEXUAL ASSAULT': 'SEXUAL OFFENSE',
    'PROSTITUTION': 'SEXUAL OFFENSE',

    'OTHER OFFENSE': 'OTHER',
    'NON-CRIMINAL': 'NON-CRIMINAL',
    'NON - CRIMINAL': 'NON-CRIMINAL',
    'NON-CRIMINAL (SUBJECT SPECIFIED)': 'NON-CRIMINAL',
}

crime_data['Crime Category'] = crime_data['Primary Type'].map(crime_mapping) # Apply mapping

data = crime_data[feature_cols + ['Crime Category']].dropna() # Group with feature columns and drop rows with NaN values

le_grouped = LabelEncoder() # Second encoding to avoid overwriting

# 3. Encode categoricals
for col in ['District', 'Ward', 'Community Area', 'Location Description']:
    data[col] = data[col].astype('category')

data['Crime_Category_Label'] = le_grouped.fit_transform(data['Crime Category']) # Label encoding
# for i, label in enumerate(le_grouped.classes_):
#     print(f"{i}: {label}")

# 4. Train/Test split
train_data = data[data['Year'] < 2020]
test_data = data[data['Year'] >= 2020]

X_train = train_data[feature_cols]
y_train = train_data['Crime_Category_Label']
X_test = test_data[feature_cols]
y_test = test_data['Crime_Category_Label']  

# 5. Train model
model = LGBMClassifier(
    objective='multiclass',
    num_class=len(le_grouped.classes_),
    class_weight='balanced',
    n_estimators=300,
    learning_rate=0.1,
    max_depth=10,
    random_state=42
)
model.fit(X_train, y_train)

# 6. Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred, target_names=le_grouped.classes_))

cm = confusion_matrix(y_test, y_pred)
# Plot confusion matrix
plt.figure(figsize=(15, 12))
sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', 
            xticklabels=le_grouped.classes_, 
            yticklabels=le_grouped.classes_)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()




# # I should have made a jupiter notebook haha


# # --- Predict 2024 Crimes (Synthetic Inference) ---

# # Step 1: Create synthetic 2024 data (one entry per hour for example)
# date_range_2024 = pd.date_range(start="2024-01-01", end="2024-12-31 23:00:00", freq="H")
# df_2024 = pd.DataFrame({'Date': date_range_2024})

# # Step 2: Extract time features
# df_2024['Year'] = df_2024['Date'].dt.year
# df_2024['Month'] = df_2024['Date'].dt.month
# df_2024['Day'] = df_2024['Date'].dt.day
# df_2024['Hour'] = df_2024['Date'].dt.hour
# df_2024['DayOfWeek'] = df_2024['Date'].dt.dayofweek
# df_2024['IsWeekend'] = df_2024['DayOfWeek'].isin([5, 6]).astype(int)

# # Step 3: Add sine/cosine cyclical features
# df_2024['Hour_sin'] = np.sin(2 * np.pi * df_2024['Hour'] / 24)
# df_2024['Hour_cos'] = np.cos(2 * np.pi * df_2024['Hour'] / 24)
# df_2024['Month_sin'] = np.sin(2 * np.pi * df_2024['Month'] / 12)
# df_2024['Month_cos'] = np.cos(2 * np.pi * df_2024['Month'] / 12)
# df_2024['DayOfWeek_sin'] = np.sin(2 * np.pi * df_2024['DayOfWeek'] / 7)
# df_2024['DayOfWeek_cos'] = np.cos(2 * np.pi * df_2024['DayOfWeek'] / 7)

# # Step 4: Compute default values from dataset
# default_values = {
#     'District': data['District'].mode()[0],            # Most common district
#     'Ward': data['Ward'].mode()[0],                    # Most common ward
#     'Community Area': data['Community Area'].mode()[0],
#     'Latitude': data['Latitude'].mean(),               # Average location
#     'Longitude': data['Longitude'].mean(),
#     'Location Description': data['Location Description'].mode()[0],
#     'Domestic': 0,
#     'Arrest': 0,
# }
# for col, value in default_values.items():
#     df_2024[col] = value


# # Step 5: Match column types
# df_2024['District'] = df_2024['District'].astype('category')
# df_2024['Ward'] = df_2024['Ward'].astype('category')
# df_2024['Community Area'] = df_2024['Community Area'].astype('category')
# df_2024['Location Description'] = df_2024['Location Description'].astype('category')

# # Step 6: Select feature columns (same order as training!)
# X_2024 = df_2024[
#     [
#     'Year', 'Month_sin', 'Month_cos', 'Day', 'Hour_sin', 
#     'Hour_cos', 'DayOfWeek_sin', 'DayOfWeek_cos', 'IsWeekend',
#     'District', 'Ward', 'Community Area','Latitude', 'Longitude',
#     'Location Description', 'Domestic', 'Arrest'
#     ]
# ]

# # Step 7: Predict with your trained model
# y_2024_preds = model.predict(X_2024)
# predicted_categories = le_grouped.inverse_transform(y_2024_preds)
# df_2024['Predicted Crime Category'] = predicted_categories

# # Step 8: Plot results on a form of daily heatmap 

# daily_counts = df_2024.groupby([df_2024['Date'].dt.date, 'Predicted Crime Category']).size().unstack().fillna(0)

# plt.figure(figsize=(18, 10))
# sns.heatmap(daily_counts.T, cmap='YlGnBu', cbar_kws={'label': 'Predicted Crime Count'})
# plt.title('Predicted Daily Crime Categories in 2024')
# plt.xlabel('Date')
# plt.ylabel('Crime Category')
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()

# # Next step : Use random samples from the original distribution instead of static values.
# # And maybe generate separate 2024 prediction batches for different locations or scenario testing.
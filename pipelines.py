
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

df = pd.read_csv(r"C:/Users/krith/OneDrive/Desktop/Virat/71 Centuries of Virat Kohli.csv")

df.shape

df.columns.tolist()

df.head()

df["Date"]=pd.to_datetime(df['Date'],format="%d/%m/%Y")

df["Date"]

df["Year"]=df["Date"].dt.year

df["Month"]=df["Date"].dt.month

df.head()

df["Captain?"]=(df['Captain']=='Yes').astype(int)

df["Home?"]=(df['H/A']=='Home').astype(int)

df['Notout'] = (df['Out/Not Out'] == 'Not Out').astype(int)
df['MOM?'] = (df['Man of the Match'] == 'Yes').astype(int)

df['Won?'] = df['Result'].apply(lambda x: 1 if x == 'Won' else 0)

df.head()

df['Strike Rate'] = pd.to_numeric(df['Strike Rate'], errors='coerce')
mean_strike_rates = df.groupby('Format')['Strike Rate'].transform(lambda x: x.mean())
df['Strike Rate'] = df['Strike Rate'].fillna(mean_strike_rates)

df.describe()

df.head()

df.columns.tolist()

format_counts=df["Format"].value_counts()
print(format_counts)

opponent_counts=df["Against"].value_counts()
print(opponent_counts)

result_counts = df['Result'].value_counts()
print(result_counts)

plt.figure(figsize=(12,6))

plt.figure(figsize=(12,6))
sns.countplot(x="Format",data=df)
plt.title("Number of Centuries By Format")
plt.xlabel('Format')
plt.ylabel('Count')

plt.figure(figsize=(12,6))
sns.histplot(x="Score",data=df,hue="Format",bins=20,multiple='stack')
plt.xlabel('Score')
plt.ylabel('Frequency')

plt.figure(figsize=(16,8))
sns.scatterplot(x="Date",y="Score",hue="Format",size="Notout",sizes=(50,200),data=df)
plt.xlabel('Date')
plt.ylabel('Score')
plt.xticks(rotation=45)

plt.figure(figsize=(14,8))
top_opponents=opponent_counts.nlargest(6).index
df_top_opponents=df[df['Against'].isin(top_opponents)]
sns.boxplot(x="Against",y="Score",data=df_top_opponents)
plt.xlabel('Opponent')
plt.ylabel('Score')
plt.xticks(rotation=45)

venue_counts=df['Venue'].value_counts().nlargest(10)
print(venue_counts)

df.head()

numerical_features=['Batting Order','Inn.','Year','Month','Captain?','Home?','Notout','MOM?']

df['Batting Order']=pd.to_numeric(df['Batting Order'])
df['Inn.']=pd.to_numeric(df['Inn.'])

categorical_features=['Against','Venue','Format']

numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

X=df[numerical_features+categorical_features]
y_score=df['Score']

score_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

X_train_score, X_test_score, y_train_score, y_test_score = train_test_split(X, y_score, test_size=0.2, random_state=42)

print("\nTraining the Score Prediction Model...")
score_pipeline.fit(X_train_score, y_train_score)

y_pred_score = score_pipeline.predict(X_test_score)

print("\nScore Prediction Model Evaluation:")
mse = mean_squared_error(y_test_score, y_pred_score)
r2 = r2_score(y_test_score, y_pred_score)
print(f'Mean Squared Error: {mse:.2f}')
print(f'R² Score: {r2:.2f}')
print(f'Root Mean Squared Error: {np.sqrt(mse):.2f}')

df['Match_Result_Binary'] = df['Result'].apply(lambda x: 1 if x == 'Won' else 0)
y_result = df['Match_Result_Binary']

X_train_result, X_test_result, y_train_result, y_test_result = train_test_split(
    X, y_result, test_size=0.2, random_state=42)

result_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

print("\nTraining the Match Result Prediction Model...")
result_pipeline.fit(X_train_result, y_train_result)

y_pred_result = result_pipeline.predict(X_test_result)

print("\nMatch Result Prediction Model Evaluation:")
accuracy = accuracy_score(y_test_result, y_pred_result)
print(f'Accuracy: {accuracy:.2f}')
print("\nClassification Report:")
print(classification_report(y_test_result, y_pred_result))
df.head()

def predict_kohli_century(opponent, venue, format_type, batting_order, innings,is_captain=True, is_home=True, is_mom=False):
  input_data = pd.DataFrame({
        'Batting Order': [batting_order],
        'Inn.': [innings],
        'Against': [opponent],
        'Venue': [venue],
        'Format': [format_type],
        'Year': [datetime.now().year],
        'Month': [datetime.now().month],
        'Captain?': [1 if is_captain else 0],
        'Home?': [1 if is_home else 0],
        'Notout': [0],
        'MOM?': [1 if is_mom else 0]
    })
  predicted_score = score_pipeline.predict(input_data)[0]
  predicted_result = result_pipeline.predict(input_data)[0]
  predicted_score = round(predicted_score)

  return {
        'predicted_score': predicted_score,
        'predicted_result': 'Win' if predicted_result == 1 else 'Not Win'
  }

print("\nExample Predictions:")
test_prediction = predict_kohli_century(
    opponent='Australia',
    venue='Melbourne Cricket Ground',
    format_type='Test',
    batting_order=4,
    innings=2,
    is_captain=True,
    is_home=False
)
print("\n1. Test Match Prediction:")
print(f"Opponent: Australia at Melbourne Cricket Ground")
print(f"Format: Test, Batting Position: 4, Innings: 2")
print(f"Predicted Score: {test_prediction['predicted_score']}")
print(f"Predicted Match Result: {test_prediction['predicted_result']}")

odi_prediction = predict_kohli_century(
    opponent='England',
    venue='Wankhede Stadium',
    format_type='ODI',
    batting_order=3,
    innings=2,
    is_captain=True,
    is_home=True
)
print("\n2. ODI Match Prediction:")
print(f"Opponent: England at Wankhede Stadium")
print(f"Format: ODI, Batting Position: 3, Innings: 2")
print(f"Predicted Score: {odi_prediction['predicted_score']}")
print(f"Predicted Match Result: {odi_prediction['predicted_result']}")

t20_prediction = predict_kohli_century(
    opponent='Pakistan',
    venue='Dubai International Cricket Stadium',
    format_type='T20I',
    batting_order=3,
    innings=1,
    is_captain=False,
    is_home=False
)
print("\n3. T20I Match Prediction:")
print(f"Opponent: Pakistan at Dubai International Cricket Stadium")
print(f"Format: T20I, Batting Position: 3, Innings: 1")
print(f"Predicted Score: {t20_prediction['predicted_score']}")
print(f"Predicted Match Result: {t20_prediction['predicted_result']}")

print("\nExample Predictions:")
test_prediction = predict_kohli_century(
    opponent='Australia',
    venue='Melbourne Cricket Ground',
    format_type='Test',
    batting_order=4,
    innings=2,
    is_captain=True,
    is_home=False
)
print("\n1. Test Match Prediction:")
print(f"Opponent: Australia at Melbourne Cricket Ground")
print(f"Format: Test, Batting Position: 4, Innings: 2")
print(f"Predicted Score: {test_prediction['predicted_score']}")
print(f"Predicted Match Result: {test_prediction['predicted_result']}")

import pickle

with open('score_pipeline.pkl', 'wb') as f:
    pickle.dump(score_pipeline, f)

with open('result_pipeline.pkl', 'wb') as f:
    pickle.dump(result_pipeline, f)

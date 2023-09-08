import pandas as pd
import streamlit as st 
import pickle 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
st.set_option('deprecation.showPyplotGlobalUse', False)

tab_1,tab_2 = st.tabs(['VIEW PREDICTION','DATAFRAME AND DOWNLOAD'])
nltk.download('vader_lexicon')
model = pickle.load(open("rf_model", 'rb'))

st.sidebar.title("Data Input")
age = st.sidebar.number_input("Input the Age",18,99)
rating = st.sidebar.number_input("Input the Ratings",1,5)
review = st.sidebar.text_input("Input The Review Here","This is my review")
pfc = st.sidebar.number_input("Input the Positive Feedback Count",0,122)
cid = st.sidebar.number_input("Input the Clothing ID",0,1205)
cls = st.sidebar.selectbox("Select the Class of the Clothe",["Dresses","Knits","Blouses","Sweaters","Pants",
                                                              "Jeans","Fine gauge","Skirts","Jackets","Lounge",
                                                              "Swim","Outerwear","Shorts","Sleep","Legwear",
                                                              "Intimates","Layering","Trend","Casual bottoms",
                                                              "Chemises"])
dept = st.sidebar.selectbox("Select the Department",["Tops","Dressers","Bottoms","Intimate","Jackets","Trend"])
div = st.sidebar.selectbox("Select the Division",["General","General Petite","Intimate"])

df = pd.DataFrame()

df['Clothing ID'] = [cid]
df["Age"] = [age]
df["Review Text"] = [review]
df["Rating"] = [rating]
df['Positive Feedback Count'] = [pfc]
df["Division Name"] = [div]
df['Department Name'] = [dept]
df['Class Name'] = [cls]

sia = SentimentIntensityAnalyzer()

df['sentiment_score'] = df['Review Text'].apply(lambda x: sia.polarity_scores(str(x))['compound'])
df['sentiment_category'] = df['sentiment_score'].apply(lambda score: 'positive' if score >= 0.05 else 'negative' if score <= -0.05 else 'neutral')

df.drop("Review Text", axis = 1, inplace= True)
cat = ['Division Name', 'Department Name', 'Class Name', 'sentiment_category']

new_cols = pd.get_dummies(df[cat])

for i in new_cols.columns :
    new_cols[i] = new_cols[i].astype('int8')

encode_cols = ['Division Name_General', 'Division Name_General Petite',
       'Division Name_Initmates', 'Department Name_Bottoms',
       'Department Name_Dresses', 'Department Name_Intimate',
       'Department Name_Jackets', 'Department Name_Tops',
       'Department Name_Trend', 'Class Name_Blouses',
       'Class Name_Casual bottoms', 'Class Name_Chemises',
       'Class Name_Dresses', 'Class Name_Fine gauge', 'Class Name_Intimates',
       'Class Name_Jackets', 'Class Name_Jeans', 'Class Name_Knits',
       'Class Name_Layering', 'Class Name_Legwear', 'Class Name_Lounge',
       'Class Name_Outerwear', 'Class Name_Pants', 'Class Name_Shorts',
       'Class Name_Skirts', 'Class Name_Sleep', 'Class Name_Sweaters',
       'Class Name_Swim', 'Class Name_Trend', 'sentiment_category_negative',
       'sentiment_category_neutral', 'sentiment_category_positive']
df_encode = pd.DataFrame()
for i in (encode_cols) : 
    if i in (new_cols.columns) :
        df_encode[i] = [1]
    else :
        df_encode[i] = [0]

df = df.drop(cat,axis=1)
data = df.join(df_encode)

pred = model.predict(data)

if pred == 0 :
        prediction = "Not Recommended"
else : 
    prediction = "Recommended"

df["prediction"] = prediction

tab_1.success("Prediction")
tab_1.title(f"From the features provided, The product was {prediction} by the customer")

tab_1.success("Prediction probability")
tab_1.write(f'probability of having a recommendation is {model.predict_proba(data)[:,1] * 100} %')

tab_2.success("Dataframe after prediction")
tab_2.dataframe(df)
classes = ["Not Recommended Probability", "Recommended Probability"]
proba = model.predict_proba(data)
fig = sns.barplot(x=np.arange(len(proba[0])), y=proba[0])
plt.xticks(np.arange(len(proba[0])), labels=[f"Class {i}" for i in classes])
plt.xlabel('Recommendation Probability')
plt.ylabel('Probability')
plt.title(f'Predicted Probabilities for Single Sample')
plt.savefig('predicted_probabilities.png')
tab_2.success("Count Plot for prediction probability")
tab_2.pyplot()
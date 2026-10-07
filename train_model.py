import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

df = pd.read_csv("data/logistics_cleaned.csv")
features = ["Shipping_Mode","Market","Order_Region","Category","Customer_Segment",
            "Quantity","Product_Price","Discount_Rate","Sales","Profit_Ratio","Scheduled_Days"]
X = df[features]
y = df["Late_Flag"]

cat = ["Shipping_Mode","Market","Order_Region","Category","Customer_Segment"]
num = ["Quantity","Product_Price","Discount_Rate","Sales","Profit_Ratio","Scheduled_Days"]

pre = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
    ("num", StandardScaler(), num)
])

X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.20,random_state=42,stratify=y
)

model = Pipeline([
    ("pre",pre),
    ("model",RandomForestClassifier(
        n_estimators=250,max_depth=14,min_samples_leaf=3,
        random_state=42,class_weight="balanced"
    ))
])
model.fit(X_train,y_train)
pred = model.predict(X_test)
prob = model.predict_proba(X_test)[:,1]

print("Accuracy:", round(accuracy_score(y_test,pred),4))
print("Precision:", round(precision_score(y_test,pred),4))
print("Recall:", round(recall_score(y_test,pred),4))
print("F1:", round(f1_score(y_test,pred),4))
print("ROC-AUC:", round(roc_auc_score(y_test,prob),4))

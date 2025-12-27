import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df = pd.read_csv("projec1.csv")
print(df)
print(df.head(5))
print(df.info())
print(df.describe())

df["Result"] = df["Result"].map({"fail":0, "pass":1})
X = df[["Hours","Attendance","PrevMarks"]]
y = df["Result"]
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2
, random_state = 42)
print("\n---- TRAIN DATA ----")
print("X_train:")
print(X_train)

print("\ny_train:")
print(y_train)

print("\n---- TEST DATA ----")
print("X_test:")
print(X_test)

print("\ny_test:")
print(y_test)

model = LogisticRegression()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)
hours = float(input("Enter study hours: "))
attendance = float(input("Enter attendance: "))
prev_marks = float(input("Enter previous marks: "))
input_df = pd.DataFrame(
    [[hours, attendance, prev_marks]],
    columns=["Hours", "Attendance", "PrevMarks"]
)

Result = model.predict(input_df)[0]

if Result == 1:
    print("Student will PASS")
else:
    print("Student will FAIL")

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

df = pd.read_csv("projec1.csv")

df["Result"] = df["Result"].map({"fail": 0, "pass": 1})

X = df[["Hours", "Attendance", "PrevMarks"]]
y = df["Result"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

hours = float(input("Enter study hours: "))
attendance = float(input("Enter attendance: "))
prev_marks = float(input("Enter previous marks: "))

if hours < 0 or attendance < 0 or prev_marks < 0:
    print("Invalid input values")
else:
    input_df = pd.DataFrame(
        [[hours, attendance, prev_marks]],
        columns=["Hours", "Attendance", "PrevMarks"]
    )

    input_df = scaler.transform(input_df)
    result = model.predict(input_df)[0]

    if result == 1:
        print("Student will PASS")
    else:
        print("Student will FAIL")

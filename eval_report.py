import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np, pandas as pd, io
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

tr = pd.read_csv("data/mitbih_train.csv", header=None)
te = pd.read_csv("data/mitbih_test.csv", header=None)
print("train shape", tr.shape, "test shape", te.shape)
print("train class counts", np.bincount(tr.iloc[:, -1].astype(int)))
print("test class counts", np.bincount(te.iloc[:, -1].astype(int)))

X = te.iloc[:, :-1].values.astype("float32").reshape(-1, 187, 1)
y = te.iloc[:, -1].values.astype(int)
m = load_model("ecg_cnn_bilstm.keras")
buf = io.StringIO(); m.summary(print_fn=lambda s: buf.write(s + "\n")); print(buf.getvalue())
loss, acc = m.evaluate(X, y if False else np.eye(5)[y], verbose=0)
print("test loss", loss, "test acc", acc)
p = m.predict(X, batch_size=512, verbose=0).argmax(1)
names = ["N", "S", "V", "F", "Q"]
print(classification_report(y, p, target_names=names, digits=4))
cm = confusion_matrix(y, p)
print(cm)
print("majority baseline", (y == 0).mean())
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=names, yticklabels=names)
plt.xlabel("Predicted"); plt.ylabel("True"); plt.title("Confusion Matrix")
plt.tight_layout(); plt.savefig("Figure_Confusion_Matrix.png", dpi=200)

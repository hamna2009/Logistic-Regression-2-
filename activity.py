import numpy
import sklearn

x = numpy.arange(10).reshape(-1, 1)
y = numpy.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

model = sklearn.linear_model.LogisticRegression()
model.fit(x, y)
model.pred()

p_pred = model.predict_proba(x)
y_pred = model.predict(x)
score_ = model.score(x, y)
conf_m = sklearn.metrics.confusion_matrix(y, y_pred)
report = sklearn.metrics.classification_report(y, y_pred)

print("x:", x, sep="\n")
print("y:", y, sep="\n", end="\n\n")
print("Intercept:", model.intercept_)
print("Coef:", model.coef_, end="\n\n")
print("p_pred:", p_pred, sep="\n", end="\n\n")
print("y_pred:", y_pred, end="\n\n")








from sklearn.isotonic import IsotonicRegression
from sklearn.svm import SVR
from matplotlib import pyplot as plt
import random

raw_data = open("challange_dataset.txt").read().strip().split("\n")
x_values = [float(row.strip().split(',')[0]) for row in raw_data]
y_values = [float(row.strip().split(',')[1]) for row in raw_data]

isotonic_model = IsotonicRegression()
isotonic_model.fit(x_values, y_values)

## selecting a random index for making a prediction
random_index = random.randrange(len(x_values))

test_x_value = x_values[random_index]
expected_y_value = y_values[random_index]
predicted_y_value = isotonic_model.predict([test_x_value])[0]

print "Predicted value : " + str(predicted_y_value)
print "Expected value : " + str(expected_y_value)
print "Error : " + str(abs(predicted_y_value - expected_y_value))

plt.scatter(x_values, y_values)
plt.plot(x_values, isotonic_model.predict(x_values))
plt.show()

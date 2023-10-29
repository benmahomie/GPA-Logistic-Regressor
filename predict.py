from keras.models import load_model
import random
import matplotlib.pyplot as plt

model = load_model('models\\gpa_predict_20231024_155939.h5')

GPAS_TO_GENERATE = 50

high_school_gpa_values = [round(random.uniform(2.5, 4), 2) for _ in range(GPAS_TO_GENERATE)]

normalized_gpa_list = [gpa / 4.0 for gpa in high_school_gpa_values]

predictions = model.predict(normalized_gpa_list)

predictions = predictions * 4

print('High School GPA -> College GPA Prediction')
for i in range(len(predictions)):
    print(f'{high_school_gpa_values[i]} -> {predictions[i][0]:.2f}')

plt.scatter(high_school_gpa_values, predictions)
plt.xlabel('High School GPAs')
plt.ylabel('College GPA Predictions')
plt.show()
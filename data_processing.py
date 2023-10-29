import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from keras.callbacks import EarlyStopping
from datetime import datetime

## GLOBALS ##
MAKE_MODEL = False
HEATMAP = True
GPA_COUNTS = True

filename = 'graduation_rate.csv' # CSV data file
models = 'models' # Folder holding .h5 files

df = pd.read_csv(filename) # load dataframe

degree_mapping = {
    "some high school": 0,
    "high school": 1,
    "some college": 2,
    "associate's degree": 3,
    "bachelor's degree": 4,
    "master's degree": 5
}

df['parental level of education'] = df['parental level of education'].replace(degree_mapping)

## Process strings for correlation map
if HEATMAP:

    corr = df.corr()

    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.show()

## Count the amounts of GPAs in ranges of 0-1, 1-2, 2-3, 3-4
if GPA_COUNTS:
    bins = [1, 2, 3, 4]

    df['high_school_gpa_counts'] = pd.cut(df['high school gpa'], bins)
    hs_count = df['high_school_gpa_counts'].value_counts().sort_index()

    df['college_gpa_counts'] = pd.cut(df['college gpa'], bins)
    c_count = df['college_gpa_counts'].value_counts().sort_index()

    print(f'HS:{hs_count}\nC:{c_count}')

## ML TRAINING ##
if MAKE_MODEL:
    # X = df['high school gpa'].values
    y = df['college gpa'].values
    df2 = df.drop(columns='college gpa')
    X = df2.values

    df2['high school gpa'] = df2['high school gpa'] / 4    

    # plt.scatter(X, y)
    # plt.xlabel('High School GPAs')
    # plt.ylabel('College GPAs')
    # plt.show()

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Normalize
    y_train = y_train / 4.0
    y_test = y_test / 4.0

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(30, input_shape=[6], activation='relu'),
        tf.keras.layers.Dense(30, activation='relu'),
        tf.keras.layers.Dense(1, activation='relu')
    ])

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    early_stop = EarlyStopping(monitor='val_loss', patience=10)

    history = model.fit(
        X_train, 
        y_train, 
        epochs=1000, 
        validation_split=0.2#,  # add validation split
        # callbacks=[early_stop]  # add callbacks
    )

    loss, mae = model.evaluate(X_test, y_test)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model.save(f"{models}\\gpa_predict_{timestamp}.h5")
    print(f'Saved as gpa_predict_{timestamp}.h5')
import pandas as pd
import numpy as np
import bayesian_linear_regression.bayesian_linear_regeression as br

# Get data.
raw_data = pd.read_csv(
    r'C:\Users\mdjordjic\source\repos\glm_hyper_parameter_optimization'
    r'\glm_hyper_parameter_optimization\x64\Release\data_set.csv',
    header=None
)

# Split data.
training_data = raw_data.iloc[0:int(len(raw_data) * .8), ]
testing_data = raw_data.iloc[int(len(raw_data) * .8):, ]

# Get features (target is is placed in column no. 2).
index_of_target_column = 2
selection_vector = \
    [i != index_of_target_column for i in range(0, raw_data.shape[1])]
features_for_training = np.array(
    training_data.loc[:, selection_vector]
)
features_for_testing = np.array(
    testing_data.loc[:, selection_vector]
)

# Standardize features.
mean = np.mean(features_for_training, axis=0)
standard_deviation = np.std(features_for_training, axis=0)
x_train = (features_for_training-mean) / standard_deviation
x_test = (features_for_testing-mean) / standard_deviation
print(np.var(x_train, axis=0))
print(np.var(x_test, axis=0))

# Add bias.
x_train = np.hstack((x_train, np.ones(shape=(len(x_train), 1))))
x_test = np.hstack((x_test, np.ones(shape=(len(x_test), 1))))

# Make targets.
y_train = training_data.iloc[:, index_of_target_column].values
y_test = testing_data.iloc[:, index_of_target_column].values

# Produce diagnostics video.
errors, mll = br.sequential_fit(
    features_for_training=x_train,
    targets_for_training=y_train,
    features_for_testing=x_test,
    targets_for_testing=y_test,
    steps=25,
    video_file=r'C:\Users\mdjordjic\br.mp4',
    produce_video=True
)

# Plot diagnostics.
diagnostics = np.vstack((errors, mll))
figure, ax = plt.subplots(2, 1)
ax[0].plot(diagnostics[0, :])
ax[1].plot(diagnostics[1, :])
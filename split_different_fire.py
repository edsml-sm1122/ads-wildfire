import numpy as np


# The percentage of pixels with 1 in a whole 256 * 256
def calculate_percentage(array):
    percentages = []
    for i in range(array.shape[0]):
        count = np.sum(array[i] == 1)
        percentage = (count / array.shape[1]) * 100
        percentages.append(percentage)
    return percentages


# Calculates the difference between adjacent elements
def calculate_differences(array):
    differences = np.diff(array)
    negative_indices = np.where(differences < 0)[0]
    return differences, negative_indices


# Calculates the drop percentages that based on previous time step
def calculate_drop_percentages(indices, percentages, differences):
    drop_percentages = []
    for index in indices:
        drop_percentage = (-differences[index]) / percentages[index] * 100
        drop_percentages.append(drop_percentage)
    return drop_percentages


# find all the end index of each fire
def find_indices_above_average(indices, drop_percentages, average):
    indices_above_average = [index for index, drop_percentage in zip(indices, drop_percentages) if drop_percentage > average]
    return indices_above_average


# split into different fires, each fire is ndarray, and all the fires are saved into a list
def split_into_different_fires(data):
    # flatten the array:
    train_flattened = train_data.reshape(train_data.shape[0], -1)
    train_percentages = calculate_percentage(train_flattened)
    train_differences, train_negative_indices = calculate_differences(train_percentages)
    train_drop_percentages = calculate_drop_percentages(train_negative_indices, train_percentages, train_differences)
    average_drop_percentage = np.mean(train_drop_percentages)
    train_indices_above_average = find_indices_above_average(train_negative_indices,
                                                             train_drop_percentages,
                                                             average_drop_percentage)

    new_fire_start_index = [0] + train_indices_above_average + [len(data)]
    split_data = [data[new_fire_start_index[i]:new_fire_start_index[i + 1]] for i in
                  range(len(new_fire_start_index) - 1)]
    return split_data


if __name__ == '__main__':
    dataset_path = "./dataset/"
    train_data = np.load(dataset_path + "Ferguson_fire_train.npy")

    data_list = split_into_different_fires(train_data)
    for i in range(len(data_list)):
        print(data_list[i].shape)








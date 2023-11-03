# This is a sample Python script.

# Press Maiusc+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import numpy
from Scripts.Classifier import Classifier
from Scripts.FeatureClassifier import FeatureClassifier
from Scripts.PathManager import PathManager

numpy.random.seed(42)

# Press the green button in the gutter to run the script.

if __name__ == '__main__':

    general_path = PathManager()
    general_path.initialize_folder_path()
    print(general_path.get_complete_path())

    print("TEST DATI FORESTE 2017:\n")

    print("Acquisizione dei set di training...")
    path_image2017 = general_path.get_complete_path() + "/DatasetSentinel2/DataSetSentinel2_2017"
    path_mask = general_path.get_complete_path() + "/masks"

    x_train_2017 = FeatureClassifier()
    x_train_2017.set_x_feature(path_image2017, 0, 77)
    x_train_2017.replace_nan_value()
    x_train_2017.replace_value(255, 1)
    print(x_train_2017.get_feature().shape)

    Y_train_2017 = FeatureClassifier()
    Y_train_2017.set_y_feature(path_mask, 0, 77)
    Y_train_2017.replace_nan_value()
    Y_train_2017.replace_value(255, 1)
    y_train_2017 = Y_train_2017.flatten_array()
    print(Y_train_2017.get_feature().shape)

    print("Acquisizione dei set di test...")
    x_test_2017 = FeatureClassifier()
    x_test_2017.set_x_feature(path_image2017, 78, 93)
    x_test_2017.replace_nan_value()
    x_test_2017.replace_value(255, 1)
    print(x_test_2017.get_feature().shape)

    Y_test_2017 = FeatureClassifier()
    Y_test_2017.set_y_feature(path_mask, 78, 93)
    Y_test_2017.replace_nan_value()
    Y_test_2017.replace_value(255, 1)
    print(Y_test_2017.get_feature().shape)
    y_test_2017 = Y_test_2017.flatten_array()

    print("Addestramento e classificazione per i set di training e di test...")
    clf_2017 = Classifier()
    prediction = clf_2017.classify(x_train_2017.get_feature(), y_train_2017, x_test_2017.get_feature())

    count_0 = 0
    count_1 = 0
    for item in prediction:
        if item == 1:
            count_1 += 1
        if item == 0:
            count_0 += 1
    print("\n\nI risultati derivati dalla prediction sono:\n")
    print(f"    (i)  Gli uno sono: {count_1};\n"
          f"    (ii) Gli zero sono: {count_0}.")

    clf_2017.print_confusion_matrix(y_test_2017, prediction)
    clf_2017.print_classification_report(y_test_2017, prediction)

    print("TEST DATI FORESTE 2018:\n")

    print("Acquisizione dei set di training...")
    path_image2018 = general_path.get_complete_path() + "/DatasetSentinel2/DataSetSentinel2_2018"
    path_mask = general_path.get_complete_path() + "/masks"

    x_train_2018 = FeatureClassifier()
    x_train_2018.set_x_feature(path_image2018, 0, 77)
    x_train_2018.replace_nan_value()
    x_train_2018.replace_value(255, 1)
    print(x_train_2018.get_feature().shape)

    Y_train_2018 = FeatureClassifier()
    Y_train_2018.set_y_feature(path_mask, 0, 77)
    Y_train_2018.replace_nan_value()
    Y_train_2018.replace_value(255, 1)
    y_train_2018 = Y_train_2018.flatten_array()
    print(Y_train_2018.get_feature().shape)

    print("Acquisizione dei set di test...")
    x_test_2018 = FeatureClassifier()
    x_test_2018.set_x_feature(path_image2018, 78, 93)
    x_test_2018.replace_nan_value()
    x_test_2018.replace_value(255, 1)
    print(x_test_2018.get_feature().shape)

    Y_test_2018 = FeatureClassifier()
    Y_test_2018.set_y_feature(path_mask, 78, 93)
    Y_test_2018.replace_nan_value()
    Y_test_2018.replace_value(255, 1)
    print(Y_test_2018.get_feature().shape)
    y_test_2018 = Y_test_2018.flatten_array()

    print("Addestramento e classificazione per i set di training e di test...")
    clf_2018 = Classifier()
    prediction = clf_2018.classify(x_train_2018.get_feature(), y_train_2018, x_test_2018.get_feature())

    count_0 = 0
    count_1 = 0
    for item in prediction:
        if item == 1:
            count_1 += 1
        if item == 0:
            count_0 += 1
    print("\n\nI risultati derivati dalla prediction sono:\n")
    print(f"    (i)  Gli uno sono: {count_1};\n"
          f"    (ii) Gli zero sono: {count_0}.")

    clf_2018.print_confusion_matrix(y_test_2018, prediction)
    clf_2018.print_classification_report(y_test_2018, prediction)

    print("TEST DATI FORESTE 2017-2018:\n")

    print("Acquisizione dei set di training...")
    x_train = numpy.concatenate([x_train_2017.get_feature(), x_train_2018.get_feature()], axis=1)
    y_train = y_train_2018
    print(x_train.shape)
    print(y_train.shape)
    print("Acquisizione dei set di test...")
    x_test = numpy.concatenate([x_test_2017.get_feature(), x_test_2018.get_feature()], axis=1)
    y_test = y_test_2018
    print(x_test.shape)
    print(y_test.shape)

    print("Addestramento e classificazione per i set di training e di test...")
    clf = Classifier()
    prediction = clf.classify(x_train, y_train, x_test)

    count_0 = 0
    count_1 = 0
    for item in prediction:
        if item == 1:
            count_1 += 1
        if item == 0:
            count_0 += 1
    print("\n\nI risultati derivati dalla prediction sono:\n")
    print(f"    (i)  Gli uno sono: {count_1};\n"
          f"    (ii) Gli zero sono: {count_0}.")

    clf.print_confusion_matrix(y_test, prediction)
    clf.print_classification_report(y_test, prediction)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

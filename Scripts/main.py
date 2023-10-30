# This is a sample Python script.

# Press Maiusc+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

from Scripts.Classifier import Classifier
from Scripts.FeatureClassifier import FeatureClassifier

# Press the green button in the gutter to run the script.

if __name__ == '__main__':

    print("Acquisizione dei set di training...")
    x_train = FeatureClassifier()
    x_train.set_x_feature(0, 77)
    x_train.replace_nan_value()
    x_train.replace_value(255, 1)

    Y_train = FeatureClassifier()
    Y_train.set_y_feature(0, 77)
    Y_train.replace_nan_value()
    Y_train.replace_value(255, 1)
    y_train = Y_train.flatten_array()

    print("Acquisizione dei set di test...")
    x_test = FeatureClassifier()
    x_test.set_x_feature(78, 93)
    x_test.replace_nan_value()
    x_test.replace_value(255, 1)

    Y_test = FeatureClassifier()
    Y_test.set_y_feature(78, 93)
    Y_test.replace_nan_value()
    Y_test.replace_value(255, 1)
    y_test = Y_test.flatten_array()

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

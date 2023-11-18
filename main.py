# This is a sample Python script.

# Press Maiusc+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import numpy
from Scripts.Classifier.Classifier import Classifier
from Scripts.Feature.FeatureClassifier import FeatureClassifier
from Scripts.Feature.FeatureTypeException import FeatureTypeException
from Scripts.PathManager import PathManager

TYPE_FEATURE_INPUT = "input"
TYPE_FEATURE_TARGET = "target"
numpy.random.seed(42)

# Press the green button in the gutter to run the script.

if __name__ == '__main__':

    general_path = PathManager()
    general_path.initialize_folder_path()
    print(general_path.get_complete_path())

    try:
        print("TEST DATI FORESTE 2018:\n")

        print("Acquisizione dei set di training 2018...")
        path_image2018 = general_path.get_complete_path() + "/DatasetSentinel2/DataSetSentinel2_2018"
        path_mask = general_path.get_complete_path() + "/masks"

        x_train_2018 = FeatureClassifier(TYPE_FEATURE_INPUT)
        x_train_2018.set_feature(path_image2018, 0, 77)
        x_train_2018.replace_nan_value()
        x_train_2018.replace_value(255, 1)
        print(x_train_2018.get_feature().shape)

        Y_train_2018 = FeatureClassifier(TYPE_FEATURE_TARGET)
        Y_train_2018.set_feature(path_mask, 0, 77)
        Y_train_2018.replace_nan_value()
        Y_train_2018.replace_value(255, 1)
        y_train_2018 = Y_train_2018.flatten_array()
        print(Y_train_2018.get_feature().shape)

        print("Acquisizione dei set di test 2018...")
        x_test_2018 = FeatureClassifier(TYPE_FEATURE_INPUT)
        x_test_2018.set_feature(path_image2018, 78, 93)
        x_test_2018.replace_nan_value()
        x_test_2018.replace_value(255, 1)
        print(x_test_2018.get_feature().shape)

        Y_test_2018 = FeatureClassifier(TYPE_FEATURE_TARGET)
        Y_test_2018.set_feature(path_mask, 78, 93)
        Y_test_2018.replace_nan_value()
        Y_test_2018.replace_value(255, 1)
        print(Y_test_2018.get_feature().shape)
        y_test_2018 = Y_test_2018.flatten_array()

        print("Addestramento e classificazione per i set di training e di test...")
        try:
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
        except ValueError:
            print("Impossibile eseguire la classificazione per i dati 2018.\n"
                  "La funzione riceve un argomento che ha il tipo corretto ma un valore inappropriato")

        print("TEST DATI FORESTE 2017-2018:\n")

        print("Acquisizione dei set di training 2017...")
        path_image2017 = general_path.get_complete_path() + "/DatasetSentinel2/DataSetSentinel2_2017"
        path_mask = general_path.get_complete_path() + "/masks"

        x_train_2017 = FeatureClassifier(TYPE_FEATURE_INPUT)
        x_train_2017.set_feature(path_image2017, 0, 77)
        x_train_2017.replace_nan_value()
        x_train_2017.replace_value(255, 1)
        print(x_train_2017.get_feature().shape)

        Y_train_2017 = FeatureClassifier(TYPE_FEATURE_TARGET)
        Y_train_2017.set_feature(path_mask, 0, 77)
        Y_train_2017.replace_nan_value()
        Y_train_2017.replace_value(255, 1)
        y_train_2017 = Y_train_2017.flatten_array()
        print(Y_train_2017.get_feature().shape)

        print("Acquisizione dei set di test 2017...")
        x_test_2017 = FeatureClassifier(TYPE_FEATURE_INPUT)
        x_test_2017.set_feature(path_image2017, 78, 93)
        x_test_2017.replace_nan_value()
        x_test_2017.replace_value(255, 1)
        print(x_test_2017.get_feature().shape)

        Y_test_2017 = FeatureClassifier(TYPE_FEATURE_TARGET)
        Y_test_2017.set_feature(path_mask, 78, 93)
        Y_test_2017.replace_nan_value()
        Y_test_2017.replace_value(255, 1)
        print(Y_test_2017.get_feature().shape)
        y_test_2017 = Y_test_2017.flatten_array()

        print("Concatenazione dei set di training 2017 e 2018...")
        x_train = numpy.concatenate([x_train_2017.get_feature(), x_train_2018.get_feature()], axis=1)
        y_train = y_train_2018
        print(x_train.shape)
        print(y_train.shape)
        print("Concatenazione dei set di test 2017 e 2018...")
        x_test = numpy.concatenate([x_test_2017.get_feature(), x_test_2018.get_feature()], axis=1)
        y_test = y_test_2018
        print(x_test.shape)
        print(y_test.shape)

        print("Addestramento e classificazione per i set di training e di test...")
        try:
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
        except ValueError:
            print("Impossibile eseguire la classificazione per i dati concatenati 2017/2018.\n"
                  "La funzione riceve un argomento che ha il tipo corretto ma un valore inappropriato")
    except FeatureTypeException:
        print("Una feature può essere solo di tipo 'input' o 'target'.\n"
              "Inserire i valori corretti e riavviare il programma")



# See PyCharm help at https://www.jetbrains.com/help/pycharm/

import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report

RANDOM_STATE = 42


class Classifier:

    # Dichiarazione variabili

    __clf: RandomForestClassifier

    """
    Nome: __init__

    Input: /

    Output: /

    Comportamento: Inizializza la variabile __clf
    """
    def __init__(self):
        self.__clf = None

    """
    Nome: __train

    Input: feature di attestramento x_train e y_train di tipo FeatureClassifier

    Output: /

    Comportamento: addestra il classificatore con le variabili passate in input
    """
    def __train(self, x_train, y_train):
        self.__clf = RandomForestClassifier(class_weight={0: 1, 1: 10}, random_state=RANDOM_STATE)
        self.__clf.fit(x_train.get_feature(), y_train)

    """
    Nome: __prediction

    Input: feature x_test di tipo FeatureCollection

    Output: classi di previsione come array

    Comportamento: restituisce le classi di predizione date dalla funzione predict del RandomForestClassifier
    """
    def __prediction(self, x_test):
        prediction = self.__clf.predict(x_test.get_feature())
        return prediction

    """
    Nome: classify

    Input: feature di addestramento x_train e y_train e feature di test x_test tutti di tipo FeatureClassifier

    Output: classi di previsione come array

    Comportamento: effettua la classificazione restituendo i rsultati della previsione del classificatore
    """
    def classify(self, x_train, y_train, x_test):
        self.__train(x_train, y_train)
        prediction = self.__prediction(x_test)
        return prediction

    """
    Nome: confusionmatrix

    Input: y_true e la classe di previsione

    Output: matrice di confusione

    Comportamento: restituisce la matrice di confusione
    """
    @staticmethod
    def confusionmatrix(y_true, y_pred):
        try:
            return confusion_matrix(y_true, y_pred)
        except sklearn.utils._param_validation.InvalidParameterError:
            print("Errore, parametri passati alla funzione confusionmatrix errati.\n"
                  "I parametri passati in input devono essere vettori a una dimensione.\n")

    """
    Nome: report

    Input: y_true e la classe di previsione

    Output: stringa contenente la precision, il recall, f1_score

    Comportamento: restituisce come stringa la precision, il recall, f1_score
    """
    @staticmethod
    def report(y_true, y_pred):
        try:
            return classification_report(y_true, y_pred)
        except sklearn.utils._param_validation.InvalidParameterError:
            print("Errore, parametri passati alla funzione report errati.\n"
                  "I parametri passati in input devono essere vettori a una dimensione.\n")


    """
    Nome: print_confusion_matrix

    Input: y_true, classe di previsione

    Output: /

    Comportamento: stampa a video la matrice di confusione
    """
    def print_confusion_matrix(self, y_true, y_pred):
        print("La matrice di confusione è:\n")
        print(self.confusionmatrix(y_true, y_pred))

    """
    Nome: print_classification_report

    Input: y_true, classe di previsione

    Output: /

    Comportamento: stampa a video il report della classificazione
    """
    def print_classification_report(self, y_true, y_pred):
        print("Il report della classificazione è:")
        print(self.report(y_true, y_pred))

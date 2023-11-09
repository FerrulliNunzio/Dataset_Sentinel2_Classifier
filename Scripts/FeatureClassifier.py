import numpy
from numpy import ndarray
from Scripts.TiffFileManager import TiffFileManager

FILE_EXTENSION = ".tif"


class FeatureClassifier:
    # Dichiarazione variabili

    __feature: ndarray

    """
    Nome: __init__

    Input: /

    Output: /

    Comportamento: inizializza la variabile __feature
    """
    def __init__(self):
        self.__feature = None

    """
    Nome: set_y_feature

    Input: elemento da cui partire (count_start) e l'elemento dove fermarsi (count_finish)

    Output: /

    Comportamento: avvalora la variabile __feature
    """
    def set_y_feature(self, file_path: str, count_start: int, count_finish: int):
        feature_series = []
        count = count_start
        while count <= count_finish:
            path = file_path + "/mask_" + str(count) + FILE_EXTENSION
            if TiffFileManager.file_exist(path):
                image = TiffFileManager()
                image.read_file(path)
                image_reshape = image.reshape_file()
                feature_series.append(image_reshape)
            count += 1
        self.concatenate_feature_series(feature_series)

    """
    Nome: set_x_feature

    Input: elemento da cui partire (count_start) e l'elemento dove fermarsi (count_finish)

    Output: /

    Comportamento: avvalora la variabile __feature
    """
    def set_x_feature(self, file_path: str, count_start: int, count_finish: int):
        feature_series = []
        count = count_start
        while count <= count_finish:
            path = file_path + "/geojson_" + str(count) + FILE_EXTENSION
            if TiffFileManager.file_exist(path):
                image = TiffFileManager()
                image.read_file(path)
                image_reshape = image.reshape_file()
                feature_series.append(image_reshape)
            count += 1
        self.concatenate_feature_series(feature_series)

    """
    Nome: concatenate_feature_series

    Input: lista

    Output: /

    Comportamento: utilizza la funzione concatenate di numpy per concatenare gli elementi della lista passata in input
    """
    def concatenate_feature_series(self, series: list):
        self.__feature = numpy.concatenate(series)

    """
    Nome: get_feature

    Input: /

    Output: ndarray

    Comportamento: restituisce il valore della variabile __feature
    """
    def get_feature(self):
        return self.__feature

    """
    Nome: __boolean_vector

    Input: /

    Output: vettore di elementi booleani

    Comportamento: restituisce un vettore contenente per ogni elemento
                   vero se l'elemento della feature contiene valori di tipo Nan,
                   falso altrimenti 
    """
    def __boolean_vector(self):
        nan_collection = []
        for item in self.__feature:
            nan_collection.append(numpy.isnan(item))
        return nan_collection

    """
    Nome: contain_nan

    Input: /

    Output: booleano

    Comportamento: Controlla ogni riga della feature,
                   se la riga contiene elementi di tipo Nan restituisce vero,
                   altrimenti restituisce falso 
    """
    def contain_nan(self):
        nan_collection = self.__boolean_vector()

        for item in nan_collection:
            column_index = 0
            if item[column_index]:
                return True

        return False

    """
    Nome: replace_nan_value

    Input: /

    Output: /

    Comportamento: Controlla ogni elemento della feature,
                   se l'iesimo elemento è di tipo Nan sostituisce il valore con 0
    """
    def replace_nan_value(self):
        if self.contain_nan():
            bool_vector = self.__boolean_vector()
            for row_index in range(len(bool_vector)):
                for column_index in range(len(bool_vector[row_index])):
                    if bool_vector[row_index][column_index]:
                        self.__feature[row_index][column_index] = 0

    """
    Nome: replace_value

    Input: value_to_change (rappresenta il valore da cambiare)
           value_with_change (rappresenta il valore con cui sostituire il valore di value_to_change)

    Output: /

    Comportamento Controlla ogni elemento della feature,
                  se il valore dell'elemento i-esimo della feature corrisponde a value_to_change
                  il valore viene sostituito con quello di value_with_change 
    """
    def replace_value(self, value_to_change: int, value_with_change: int):
        for row_index in range(len(self.__feature)):
            for column_index in range(len(self.__feature[row_index])):
                if self.__feature[row_index][column_index] == value_to_change:
                    self.__feature[row_index][column_index] = value_with_change

    """
    Nome: flatten_array

    Input: /

    Output: array

    Comportamento: restituisce l'array come un array ad una dimensione
    """
    def flatten_array(self):
        return self.__feature.ravel()

    """
    Nome: max_value

    Input: /

    Output: array contenente i valori massimi della feature

    Comportamento: restituisce un array contenente i valori minori della feature
    """
    def max_value(self):
        return numpy.nanmax(self.__feature, axis=0)

    """
    Nome: min_value

    Input: /

    Output: array contenente i valori minori della feature

    Comportamento: restituisce un array contenente i valori minori della feature
    """
    def min_value(self):
        return numpy.nanmin(self.__feature, axis=0)

    """
    Nome: mean_value

    Input: /

    Output: array con gli elementi medi

    Comportamento: restituisce un array contenente gli elementi di valore medio della feature
    """
    def mean_value(self):
        return numpy.nanmean(self.__feature, axis=0)

    """
    Nome: print_feature_statistic

    Input: /

    Output: /

    Comportamento: stampa a video le informazioni riguardanti le statistiche della feature
    """
    def print_feature_statistic(self):
        print(f"(i)   Il valore massimo per il vettore è: {self.max_value()};")
        print(f"(ii)  Il valore minimo per il vettore è: {self.min_value()};")
        print(f"(iii) Il valore medio per il vettore è: {self.mean_value()}.")
        print("\n\n")

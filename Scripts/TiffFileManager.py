import os
import numpy
import tifffile
from numpy import ndarray


class TiffFileManager:
    # Dichiarazione variabili

    __image = ndarray

    """
    Nome: __init__
    
    Input: /
    
    Output: /
    
    Comportamento: Inizializza la variabile __image
    """
    def __init__(self):
        self.__image = None

    """
    Nome: file_exist

    Input: stringa file_path che contiene il percorso del file

    Output: booleano

    Comportamento: restituisce vero se il file nel percorso specificato esiste,
                   altrimenti restituisce falso
    """
    @staticmethod
    def file_exist(file_path):
        return os.path.isfile(file_path)

    """
    Nome: read_file

    Input: stringa file_path che contiene il percorso del file

    Output: /

    Comportamento: Controlla se il file esiste nel percorso specificato,
                   se esiste assegna alla variabile __image un immagine come ndarray
    """
    def read_file(self, file_path):
        if self.file_exist(file_path):
            self.__image = tifffile.imread(file_path)

    """
    Nome: reshape_file

    Input: /

    Output: ndarray

    Comportamento: Utilizza la funzione reshape di numpy per restituire un l'array rimodellato
    """
    def reshape_file(self):
        column = 1
        if self.tiff_shape_length() > 2:
            column = self.__image.shape[2]
        return numpy.reshape(self.__image, (self.__image.shape[0]*self.__image.shape[1], column))

    """
    Nome: tiff_shape_length

    Input: /

    Output: numero intero che rappresenta la lunghezza dell'array

    Comportamento: restituisce la lunghezza dell'array come numero intero
    """
    def tiff_shape_length(self):
        return len(self.__image.shape)

    """
    Nome: get_tiff_shape

    Input: /

    Output: ndarray

    Comportamento: restituisce il valore della variabile __image
    """
    def get_tiff_shape(self):
        return self.__image.shape

    """
    Nome: get_first_shape_element

    Input: /

    Output: numero intero

    Comportamento: restituisce il valore nella prima posizione dello shape dell'array
    """
    def get_first_shape_element(self):
        return self.__image.shape[0]

    """
    Nome: get_end_shape_element

    Input: /

    Output: numero intero

    Comportamento: restituisce l'ultimo elemento dello shape dell'array
    """
    def get_end_shape_element(self):
        return self.__image.shape[self.tiff_shape_length() - 1]

    """
    Nome: get_shape_element

    Input: numero intero index che rappresenta l'indice dell'array

    Output: i-esimo elemento dell' shape di un array come numero intero

    Comportamento: Dopo aver controllato se l'indice passato in input è minore della lunghezza dello shape
                   restituisce il valore nell'i-esima posizione dello shape
    """
    def get_shape_element(self, index):
        if index < self.tiff_shape_length():
            return self.__image.shape[index]
        return

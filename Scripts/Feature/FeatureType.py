from Scripts.Feature.FeatureTypeException import FeatureTypeException


class FeatureType:
    __Type: str

    def __init__(self, feature_type: str):
        type_input = feature_type.lower()
        if (not type_input == "input") and (not type_input == "target") :
            raise FeatureTypeException("Una feature può essere solo di tipo 'input' o 'target'.\n"
                                       "Inserire i valori corretti e riavviare il programma")
        else:
            self.__Type = feature_type.lower()

    def get_type(self):
        return self.__Type

    def equals(self, string: str):
        return self.__Type == string

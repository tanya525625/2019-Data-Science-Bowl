import hashlib

from numpy import array


def make_number_hashes_for_list(list_for_enc: list):
    """
    Function for encoding strings
    from list to the numbers

    :param list_for_enc: list for encoding
    :return: numeric representation of the list
    """

    new_list = []
    for el in list_for_enc:
        h = hashlib.sha256(str(el).encode("UTF-8"))
        h = int(h.hexdigest(), base=16)
        new_list.append(h)
    return array(new_list)

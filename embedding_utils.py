import csv


def import_embeddings(filename):
    words = {}
    with open(filename, 'rt', encoding='utf8') as csvfile:
        r = csv.reader(csvfile, delimiter=' ', quoting=csv.QUOTE_NONE)
        for row in r:
            words[row[0]] = row[1:301]
    return words
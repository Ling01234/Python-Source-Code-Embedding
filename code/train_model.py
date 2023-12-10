import pickle
from icecream import ic


def get_data():
    with open('code/test_data.pickle', 'rb') as file:
        data = pickle.load(file)  # list of code (in dict)

    return data


def get_paths(data):
    paths = []
    for code in data:
        path = code['paths']
        paths.append(path)

    return paths


if __name__ == '__main__':
    data = get_data()
    paths = get_paths(data)

import re
from collections import Counter


def find_most_frequent_words(word_list, n):
    word_counter = Counter(word_list)
    most_common = word_counter.most_common(n)
    return most_common


def find_frequency():
    file = open("ALPR\Results\Recognized.txt", "r")
    words = []
    for line in file:
        line_word = re.findall('[a-zA-Z][a-zA-Z][a-zA-Z]\s\d\d\d\d',line)
        for w in line_word:
            words.append(w)
    file.close()

    most_frequent = find_most_frequent_words(words, 5)

    with open("plates.txt", 'w') as file:
        for word, count in most_frequent:
            file.write(f'{word}: {count}\n')


    print("Most Repeated Plates: ")
    for word, count in most_frequent:
        print(f'{word}: {count}')
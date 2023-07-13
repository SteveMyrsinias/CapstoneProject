import re

def find_frequency():
    file = open("ALPR\Results\Recognized.txt", "r")
    frequent_lplate = ""
    frequency = 0
    words = []
    for line in file:
        line_word = re.findall('[a-zA-Z][a-zA-Z][a-zA-Z]\s\d\d\d\d',line)
        for w in line_word:
            words.append(w)

    for i in range(0, len(words)):
        count = 1
        for j in range(i + 1, len(words)):
            if (words[i] == words[j]):
                count = count + 1

        if (count > frequency):
            frequency = count
            frequent_lplate = words[i]

    print("Most repeated Plate: " + frequent_lplate)
    print("Frequency: " + str(frequency))
    file.close()
    return frequent_lplate


import difflib
import numpy as np 
import math
import collections
word_list_file = open("wordlist.txt","r")
word_list = []
for word in word_list_file: 
    word_list.append(word.strip()) 
#print(word_list)
word_list_file.close()


#print(difflib.get_close_matches("be",["boomshakalaka"]))

def find(dictionary, word):
    start = 0
    end = len(dictionary) - 1

    while start <= end:
        middle = (start + end)// 2
        midpoint = dictionary[middle]
        if midpoint > word:
            end = middle - 1
        elif midpoint < word:
            start = middle + 1
        else:
            return midpoint

#print(find(word_list,"-"))
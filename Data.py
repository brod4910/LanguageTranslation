from googletrans import Translator
import time

translator = Translator()

filename = "./data/english.txt"

file_content = open(filename, "r")

translatedfile = "./data/spanish.txt"

translated_content = open(translatedfile, "w")

num_lines = sum(1 for line in file_content)

file_content.close()

file_content = open(filename, "r")

list_words = list()

iteration = 0

for i in range(num_lines):

	list_words.append(file_content.readline())

	if i % 400 == 0 or i == num_lines - 1:
		translated_words = translator.translate(list_words, dest='es')
		for translation in translated_words: 
			translated_content.write(translation.text)
			translated_content.write('\n')
		list_words = list()
		print(iteration)
		iteration += 1

file_content.close()
translated_content.close()
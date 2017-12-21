"""
Parses a file and translates that file in a language of the users choice using the google translation
API. Writes the contents to the destination file given.
This file is used on the command line.

More methods may be added for extracting training data if needed.

This file is strictly for getting the required training data for our LSTM/RNN neural network

Brian Rodriguez
"""

import argparse
from google.cloud import translate

def translate_text_from_file(file, target_lang, dest_file):
    num_lines = 0
    to_be_translated = []

    for line in open(file).readlines() : num_lines += 1

    open_file = open(file, 'r')

    translator_client = translate.Client()

    translated_file = open(dest_file, 'w')

    for i in range(num_lines + 1):
        next_line = open_file.readline()
        translated_text = translator_client.translate(next_line,
            target_language=target_lang, model=translate.NMT)
        translated_file.write(next_line + ":")
        translated_file.write(translated_text['translatedText'] + '\n')
        translated_file.write("----------\n")

    open_file.close()
    translated_file.close()

def longest_word_in_En_Sp_data(file):
    num_lines = 0
    longest_word_En = ""
    longest_word_Sp = ""

    for line in open(file).readlines() : num_lines += 1
    open_file = open(file, 'r')

    for i in range(num_lines):
        next_line = open_file.readline()
        parsed_line = next_line.split(':')

        if len(longest_word_En) < len(parsed_line[0]):
            longest_word_En = parsed_line[0]

        if len(longest_word_Sp) < len(parsed_line[1]):
            longest_word_Sp = parsed_line[1]

    print("Longest word in English: %s\n" % longest_word_En, "Length: %d" % len(longest_word_En))
    print("Longest word in Spanish: %s\n" % longest_word_Sp, "Length: %d" % len(longest_word_Sp))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Translate text from file.')

    subparsers = parser.add_subparsers(dest='command')

    translate_text_from_file_parser = subparsers.add_parser('translate_text_from_file', 
        help='Enter in a file you would like to add to translate')
    translate_text_from_file_parser.add_argument('file')
    translate_text_from_file_parser.add_argument('target_lang')
    translate_text_from_file_parser.add_argument('dest_file')

    longest_words_parser = subparsers.add_parser('longest_words', 
        help='Enter file to get longest words in English & Spanish')
    longest_words_parser.add_argument('file')


    args = parser.parse_args()

    if args.command == 'translate_text_from_file':
        translate_text_from_file(args.file, args.target_lang, args.dest_file)
    elif args.command == 'longest_words':
        longest_word_in_En_Sp_data(args.file)


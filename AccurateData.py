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
        nextLine = open_file.readline()
        translated_text = translator_client.translate(nextLine,
            target_language=target_lang, model=translate.NMT)
        translated_file.write(nextLine + ":")
        translated_file.write(translated_text['translatedText'] + '\n')
        translated_file.write("----------\n")

    open_file.close()
    translated_file.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Translate text from file.')

    subparsers = parser.add_subparsers(dest='command')

    translate_text_from_file_parser = subparsers.add_parser('translate_text_from_file', 
        help='Enter in a file you would like to add to translate')
    translate_text_from_file_parser.add_argument('file')
    translate_text_from_file_parser.add_argument('target_lang')
    translate_text_from_file_parser.add_argument('dest_file')

    args = parser.parse_args()

    if args.command == 'translate_text_from_file':
        translate_text_from_file(args.file, args.target_lang, args.dest_file)


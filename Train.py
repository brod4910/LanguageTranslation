import os as os

dataEn = open('data/english.txt').read()
dataSp = open('data/masterSpanish.txt').read()


charsEn = list(set(dataEn))

charsSp = list(set(dataSp))

data_sizeEn, vocab_sizeEn, data_sizeSp, vocab_sizeSp = len(dataEn), len(charsEn), len(dataSp), len(charsSp)

print("English data has %d chars and %d unique chars" % (data_sizeEn, vocab_sizeEn))
print("Spanish data has %d chars and %d unique chars" % (data_sizeSp, vocab_sizeSp))

char_to_int_En = {ch: i for i, ch in enumerate(charsEn)}
int_to_char_En = {i: ch for i, ch in enumerate(charsEn)}

char_to_int_Sp = {ch: i for i, ch in enumerate(charsSp)}
int_to_char_Sp = {i: ch for i, ch in enumerate(charsSp)}

print(char_to_int_En)
print(int_to_char_En)

print(char_to_int_Sp)
print(int_to_char_Sp)
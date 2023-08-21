# Not using this yet, but it definitely will be used in the future
import random
# Yes, using this now
import time

# This class represents a 2-gram matrix
# It's a 2D array of numbers
# The first index is the first letter of the 2-gram
# The second index is the second letter of the 2-gram
class TwogramMatrix:
    def __init__(self,alphabet):
        # Suck in the alphabet
        self.alphabet = alphabet
        # Turn the alphabet into a list
        self.alphabetlist = list(alphabet)
        # How many letters are in the alphabet?
        self.n = len(alphabet)
        # What would the matrix look like if it were completely random?
        self.basematrix = [[1 for i in range(self.n)] for j in range(self.n)]
        # What would this look like if I used a numpy array instead?
        # self.basematrix = np.ones((self.n,self.n))
        # That's what it would look like, but we're not going to use that - yet. Maybe later.

    # Function that maps characters to their index in the alphabet
    def letter_to_num(self,letter):
        return self.alphabetlist.index(letter)
    
    # Function that turns the index in the alphabet to the corresponding character
    def num_to_letter(self,num):
        return self.alphabetlist[num]
    
    # Function that takes in a word and returns a list of numbers
    def word_to_num_list(self,word):
        return [self.letter_to_num(letter) for letter in word]
    
    # Function that takes in a list of numbers and returns a word
    def num_list_to_word(self,num_list):
        return ''.join([self.num_to_letter(num) for num in num_list])
    
    # Normally, "n-gram" refers to a sequence of n words, but it can also refer to a sequence of n letters
    # I'm going to start with 2-grams - we can worry about 3-grams and more later
    # Function that takes in a word and returns a list of 2-grams (as numbers)
    def word_to_2grams(self,word):
        # Add leading and trailing spaces
        word = ' ' + word + ' '
        # Get the length of the word
        wordlen = len(word)
        # First, we convert the word to a list of numbers
        num_list = self.word_to_num_list(word)
        # Then, we create a list of 2-tuples (pairs) of numbers
        # The first number in each pair is the current number
        # The second number in each pair is the next number
        tuples = [(num_list[i], num_list[i+1]) for i in range(wordlen - 1)]
        # We don't convert these back to letters because numbers are easier to work with
        return tuples
    
    # Takes an in-progress matrix and a word and updates the matrix
    def update_matrix(self,matrix,word):
        # Get the 2-grams from the word
        twograms = self.word_to_2grams(word)
        # For each 2-gram in the word
        for twogram in twograms:
            # Get the first letter of the 2-gram
            first = twogram[0]
            # Get the second letter of the 2-gram
            second = twogram[1]
            # Increment the value of the matrix at the corresponding position
            matrix[first][second] += 1

    # Takes in a list of words and returns a matrix that represents the frequency of each 2-gram
    def get_matrix(self,wordlist):
        # Start with the base matrix
        matrix = self.basematrix
        # For each word in the word list
        for word in wordlist:
            # Update the matrix
            self.update_matrix(matrix,word)
        # Return the matrix
        return matrix
    
    # Takes a matrix and prints a nice table
    # Well, as nice as a table can be when it's made of numbers in the console
    # A csv file would definitely be better, but I'm not going to worry about that right now
    def print_matrix(self,matrix):
        # Print the header
        print('   ',end='')
        for letter in self.alphabetlist:
            print(letter,end='   ')
        print()
        # Print the matrix
        for i in range(self.n):
            print(self.alphabetlist[i],end='  ')
            for j in range(self.n):
                print(matrix[i][j],end='   ')
            print()

    # A function that "flattens" the matrix into a list of tuples
    # The first element of the tuple is the 2-gram (as a string of letters, NOT NUMBERS)
    # The second element of the tuple is the frequency of that 2-gram
    def flatten_matrix(self,matrix):
        flat = []
        for i in range(self.n):
            for j in range(self.n):
                flat.append((self.num_to_letter(i) + self.num_to_letter(j),matrix[i][j]))
        return flat
    
    # A function that returns the n most common 2-grams in the matrix
    # The return value is a list of tuples
    # The first element of the tuple is the 2-gram (as a string of letters, NOT NUMBERS)
    # The second element of the tuple is the frequency of that 2-gram
    def most_common(self,matrix,n):
        # Flatten the matrix
        flat = self.flatten_matrix(matrix)
        # Sort the flattened matrix by frequency
        flat = sorted(flat,key=lambda x: x[1],reverse=True)
        # Return the n most common 2-grams
        return flat[0:n]
    
    # A function that returns the n least common 2-grams in the matrix
    # The return value is a list of tuples
    # The first element of the tuple is the 2-gram (as a string of letters, NOT NUMBERS)
    # The second element of the tuple is the frequency of that 2-gram
    def least_common(self,matrix,n):
        # Flatten the matrix
        flat = self.flatten_matrix(matrix)
        # Sort the flattened matrix by frequency
        flat = sorted(flat,key=lambda x: x[1])
        # Return the n least common 2-grams
        return flat[0:n]

# Take in the word list file and turn it into, well, a list of words.
def get_word_list(fname):
    with open(fname) as f:
        word_list = f.read().splitlines()
    return word_list

# Load the word list from the file
wordlist = get_word_list('wordlist.txt')

# String of all the letters in the alphabet, plus a space at the start
# This is used to generate the "base" matrix
alphabet = ' abcdefghijklmnopqrstuvwxyz'
# The space is there because we want to be able to generate spaces/start/end of words

# Use the alphabet to create the matrix
testmatrix = TwogramMatrix(alphabet)

# A quick test to make sure the functions work
print(wordlist[0:10])
print(testmatrix.letter_to_num('a'))
print(testmatrix.num_to_letter(1))
print(testmatrix.word_to_num_list('hello'))
# Okay, those work. Leaving them commented out for now.

# That's some of the bookkeeping stuff out of the way.
# Oh, who the hell am I kidding, there's going to be a lot more bookkeeping.
# But for now, here's some fun stuff

# A quick test to make sure the function works
print(testmatrix.word_to_2grams('hello'))

# Some words to train the matrix on
# words = ["hello","world"]
# Commented that out because now we're going to use the whole word list
words = wordlist

starttime = time.time()

# Get the matrix for those words
newmatrix = testmatrix.get_matrix(words)

# How long did that take?
print(time.time() - starttime)

# How does printing look?
testmatrix.print_matrix(newmatrix)

# How about the ten most common 2-grams?
print(testmatrix.most_common(newmatrix,10))

# How about the hundred least common 2-grams?
print(testmatrix.least_common(newmatrix,100))
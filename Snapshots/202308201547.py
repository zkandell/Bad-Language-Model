import random
import time
import numpy as np

# This class represents an n-gram matrix
# It's an n-dimensional array of numbers
# The first index is the first letter of the n-gram
# And so on and so forth
class NgramMatrix:
    def __init__(self,alphabet,n):
        # Suck in the alphabet, which is going to be a list of characters
        # Not a string, because that is making assumptions about the alphabet
        # Namely, that all "tokens" are single characters
        self.alphabetlist = alphabet
        # How many letters are in the alphabet?
        self.lenalpha = len(alphabet)
        # How many letters are in each n-gram?
        self.n = n
        # Create the base matrix, an n-dimensional array of ones
        # The dimensions are all the same as the length of the alphabet
        self.basematrix = np.ones([self.lenalpha for i in range(self.n)],dtype=int)
        # Create the cumulative matrix, which is the matrix we're actually going to use
        # It starts out as a copy of the base matrix
        # As we train the matrix, we'll update the cumulative matrix
        self.cumulativematrix = self.basematrix.copy()
        # The normalized matrix is the cumulative matrix, but with each row normalized
        # This is used to generate random words
        self.normalizedmatrix = self.normalize_matrix(self.cumulativematrix)

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
    # I don't think this is actually used anywhere, but I'm leaving it in for now
    def num_list_to_word(self,num_list):
        return ''.join([self.num_to_letter(num) for num in num_list])
    
    # Adds leading and trailing spaces to a word
    # This is used to make sure that the first and last letters of a word are treated as the first and last letters of a word
    # There need to be n-1 spaces at the start and end of the word
    def pad_word(self,word):
        return ' ' * (self.n - 1) + word + ' ' * (self.n - 1)

    # Normally, "n-gram" refers to a sequence of n words, but it can also refer to a sequence of n letters
    # This function takes in a word and returns a list of n-grams (as numbers)
    def word_to_ngrams(self,word):
        # Pad the word
        word = self.pad_word(word)
        # Get the length of the word
        wordlen = len(word)
        # First, we convert the word to a list of numbers
        num_list = self.word_to_num_list(word)
        # Then, we create a list of n-tuples of numbers
        tuples = [tuple(num_list[i:i+self.n]) for i in range(wordlen - self.n + 1)]
        # We don't convert these back to letters because numbers are easier to work with
        return tuples
    
    # Takes an in-progress matrix and a word and updates the matrix
    def update_matrix(self,matrix,word):
        # Get the n-grams from the word
        ngrams = self.word_to_ngrams(word)
        # For each n-gram in the word
        for ngram in ngrams:
            # Add 1 to the value of the matrix at the corresponding position
            matrix[ngram] += 1

    # Takes in a list of words and trains the matrix on those words
    def train_matrix(self,wordlist):
        # For each word in the word list
        for word in wordlist:
            # Update the cumulative matrix
            self.update_matrix(self.cumulativematrix,word)
        # Update the normalized matrix
        self.normalizedmatrix = self.normalize_matrix(self.cumulativematrix)
        # Return the cumulative matrix
        return self.cumulativematrix
    
    # Since the matrix is an n-dimensional array, it's not easy to print
    # So I'm not going to bother with that for now
    def print_matrix(self,matrix):
        pass

    def export_matrix(self,matrix,filename):
        pass

    # A function that flattens one dimension of the matrix
    # This is used to flatten the entire matrix into a list of tuples
    # This function is recursive to handle an arbitrary number of dimensions
    # And when I've actually figured out how to do that, I'll come back and implement it
    def flatten_somehow(self):
        pass

    # A function that "flattens" the matrix into a list of tuples
    # The first element of the tuple is the n-gram (as a string of letters, NOT NUMBERS)
    # The second element of the tuple is the frequency of that n-gram
    def flatten_matrix(self,matrix):
        # Initialize the flat list
        flat = []
        

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
    
    # A function that normalizes a matrix
    # Each row is normalized so that the sum of the row is 1
    # These end up being the weights used to generate random words
    def normalize_matrix(self,matrix):
        # Get the sum of each row
        rowsums = np.sum(matrix,axis=-1)
        # Divide each row by its sum
        normalized = matrix / rowsums[:,np.newaxis]
        # Return the normalized matrix
        return normalized

# This class represents a 2-gram matrix
# It's a 2D array of numbers
# The first index is the first letter of the 2-gram
# The second index is the second letter of the 2-gram
class TwogramMatrix:
    def __init__(self,alphabet):
        # Suck in the alphabet, which is going to be a list of characters
        # Not a string, because that is making assumptions about the alphabet
        # Namely, that all "tokens" are single characters
        self.alphabetlist = alphabet
        # How many letters are in the alphabet?
        self.n = len(alphabet)
        # What would the matrix look like if it were completely random?
        self.basematrix = [[1 for i in range(self.n)] for j in range(self.n)]
        # What would this look like if I used a numpy array instead?
        # self.basematrix = np.ones((self.n,self.n))
        # That's what it would look like, but we're not going to use that - yet. Maybe later.
        # The cumulative matrix is the matrix that we're actually going to use
        # It starts out as a copy of the base matrix
        # As we train the matrix, we'll update the cumulative matrix
        self.cumulativematrix = self.basematrix.copy()
        # The normalized matrix is the cumulative matrix, but with each row normalized
        # This is used to generate random words
        self.normalizedmatrix = self.normalize_matrix(self.cumulativematrix)

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
    def train_matrix(self,wordlist):
        # For each word in the word list
        for word in wordlist:
            # Update the cumulative matrix
            self.update_matrix(self.cumulativematrix,word)
        # Update the normalized matrix
        self.normalizedmatrix = self.normalize_matrix(self.cumulativematrix)
        # Return the cumulative matrix
        return self.cumulativematrix
    
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

    # Takes a matrix and exports it to a csv file with the given filename
    def export_matrix(self,matrix,filename):
        # Open the file
        with open(filename,'w') as f:
            # Write the header
            f.write(','.join(self.alphabetlist) + '\n')
            # Write the matrix
            for i in range(self.n):
                f.write(self.alphabetlist[i] + ',')
                f.write(','.join([str(matrix[i][j]) for j in range(self.n)]) + '\n')

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
    
    # A function that normalizes a matrix
    # Each row is normalized so that the sum of the row is 1
    # These end up being the weights used to generate random words
    def normalize_matrix(self,matrix):
        # Get the sum of each row
        rowsums = [sum(row) for row in matrix]
        # Divide each row by its sum
        normalized = [[matrix[i][j]/rowsums[i] for j in range(self.n)] for i in range(self.n)]
        # Return the normalized matrix
        return normalized
    
    # A function that randomly chooses the next letter based on the normalized matrix
    # This is called by the generate_word function to keep getting new letters
    def choose_next_letter(self,letter):
        # Convert the letter to a number
        num = self.letter_to_num(letter)
        # Get the row of the normalized matrix corresponding to the given letter
        row = self.normalizedmatrix[num]
        # Choose a random number between 0 and 1
        r = random.random()
        # Initialize the cumulative probability
        cumprob = 0
        # For each letter in the alphabet
        for i in range(self.n):
            # Add the probability of the letter to the cumulative probability
            cumprob += row[i]
            # If the cumulative probability is greater than the random number
            if cumprob > r:
                # Return the letter
                return self.num_to_letter(i)
        # If we get here, something went wrong
        # I don't know what, but something
        # So raise an error
        raise Exception('Something went wrong in choose_next_letter')

    # A function that generates a random word
    # The word is generated using the normalized matrix
    # The normalized matrix is a list of lists of weights
    # The weights are the probability of each letter following the previous letter
    # start is the string the user wants the word to start with
    def generate_word(self,start=None):
        if start == None:
            # If no start string is given, start with a space
            start = ' '
        # Create a variable for the word
        word = start
        nextletter = None
        # While the next letter isn't a space
        while nextletter != ' ':
            # Get the last letter of the word
            lastletter = word[-1]
            # Get the next letter
            nextletter = self.choose_next_letter(lastletter)
            # Add the next letter to the word
            word += nextletter
        # Strip out the leading and trailing spaces
        word = word.strip()
        # Return the word
        return word

# Take in the word list file and turn it into, well, a list of words.
def get_word_list(fname):
    with open(fname) as f:
        word_list = f.read().splitlines()
    # If there are multiple words on the same line, split them up
    word_list = [word for line in word_list for word in line.split()]
    # Lowercase all the words
    word_list = [word.lower() for word in word_list]
    # Strip out all non-alphabetic characters
    word_list = [''.join([char for char in word if char.isalpha()]) for word in word_list]
    return word_list

# String of all the letters in the alphabet, plus a space at the start
# This is used to generate the "base" matrix
alphabet = ' abcdefghijklmnopqrstuvwxyz'
# The space is there because we want to be able to generate spaces/start/end of words
# Convert the alphabet string to a list of characters
alphabet = [char for char in alphabet]

# Use the alphabet to create the matrix
# testmatrix = TwogramMatrix(alphabet)
# To be figured out later:
testmatrix = NgramMatrix(alphabet,2)
# That will be used to generate 3-grams, 4-grams, etc.
# But the first test of it will be to recreate the 2-gram matrix without changing anything below
# Wish me luck on that endeavor

# Lots of whitespace is here completely by intention
# Delineates the classes and functions from the actual code
# This way, I can't miss it
# And makes it clearer where to leave things alone
# In theory, once I get the NgramMatrix class working, everything below this line should be the same
# All I need to do is change which testmatrix line above is commented out
# Everything below should remain untouched and still work
# ...in theory


















# Load the word list from the file
wordlist = get_word_list('wordlist.txt')

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
# Despite what I said above, this one has to be changed to use ngrams instead of 2grams
print(testmatrix.word_to_ngrams('hello'))

# Some words to train the matrix on
# words = ["hello","world"]
# Commented that out because now we're going to use the whole word list
words = wordlist

starttime = time.time()

# Train the matrix on the words
testmatrix.train_matrix(words)

# How long did that take?
print(time.time() - starttime)

# How does printing look?
testmatrix.print_matrix(testmatrix.cumulativematrix)

# How about printing the normalized matrix?
testmatrix.print_matrix(testmatrix.normalizedmatrix)

# How about exporting the matrix to a csv file?
# testmatrix.export_matrix(testmatrix.cumulativematrix,'testmatrix.csv')

# Now what about the normalized matrix?
# testmatrix.export_matrix(testmatrix.normalizedmatrix,'testmatrix_normalized.csv')

# How about the ten most common 2-grams?
print(testmatrix.most_common(testmatrix.cumulativematrix,10))

# How about the hundred least common 2-grams?
print(testmatrix.least_common(testmatrix.cumulativematrix,100))

# Now, let's generate 10 random words
for i in range(10):
    print(testmatrix.generate_word())
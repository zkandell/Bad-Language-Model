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
        # A dictionary that can hold arbitrary matrices
        # This is used to keep track of matrices that have been trained on different word lists
        # The key is the name of the matrix
        # The value is the matrix itself
        self.matrices = {'cumulative':self.cumulativematrix,'normalized':self.normalizedmatrix}
        # Note to self: it might not make sense to have the normalized matrix in here
        # It might be better to normalize matrices as needed
        # Or even a separate matrix class to store the matrix and its normalized form
        # I'll figure that out later

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
    # There need to be n-1 spaces at the start and one space at the end of the word
    # endspaces can be set to False to not add the space at the end of the word
    def pad_word(self,word,endspaces=True):
        # Strip spaces from the start and end of the word
        word = word.strip()
        # Add n-1 spaces to the start of the word
        # And one space to the end of the word
        return ' ' * (self.n - 1) + word + ' ' * (1 if endspaces else 0)

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
    
    # Adds a matrix to the dictionary of matrices
    # The key is the name of the matrix
    # By default, the matrix is the base matrix
    def add_matrix(self,name,matrix=None):
        # If no matrix is given, use the base matrix
        if matrix == None:
            matrix = self.basematrix
        # Add a copy of the matrix to the dictionary
        self.matrices[name] = matrix.copy()
        # Return the matrix
        return matrix
    
    # Takes an in-progress matrix and a word and updates the matrix
    def update_matrix(self,word,matrix=None):
        # If no matrix is given, use the cumulative matrix
        if matrix is None:
            matrix = self.cumulativematrix
        # Get the n-grams from the word
        ngrams = self.word_to_ngrams(word)
        # For each n-gram in the word
        for ngram in ngrams:
            # If the matrix being updated is not the cumulative matrix
            # Then you update the named matrix
            if matrix is not self.cumulativematrix:           
                matrix[ngram] += 1
            # You ALWAYS update the cumulative matrix
            self.cumulativematrix[ngram] += 1

    # Takes in a list of words and trains the matrix on those words
    def train_matrix(self,wordlist,matrix=None):
        # If no matrix is given, use the cumulative matrix
        if matrix is None:
            matrix = self.cumulativematrix
        # For each word in the word list
        for word in wordlist:
            # Update the cumulative matrix
            self.update_matrix(word,matrix)
        # Update the normalized matrix
        self.normalizedmatrix = self.normalize_matrix(self.cumulativematrix)
        # Return the cumulative matrix
        return self.cumulativematrix
    
    # Since the matrix is an n-dimensional array, it's not easy to print
    # So I'm not going to bother with that for now
    def print_matrix(self,matrix):
        pass

    # Not going to bother with this one either
    def export_matrix(self,matrix,filename):
        pass

    # A function that "flattens" the matrix into a list of tuples
    # The first element of the tuple is the n-gram (as a string of letters, NOT NUMBERS)
    # The second element of the tuple is the frequency of that n-gram
    def flatten_matrix(self,matrix):
        # Initialize the flat list
        flat = []
        # Enumerate the matrix
        for index,value in np.ndenumerate(matrix):
            ngram = self.num_list_to_word(index)
            # Add the index and value to the flat list
            flat.append((ngram,value))
        # Return the flat list
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
        # Normalize the array along the last axis
        # This is the axis that corresponds to the last letter of the n-gram
        # This is the axis that we want to sum over
        norm = np.sum(matrix,axis=-1,keepdims=True)
        # Divide the matrix by the norm
        normalized = matrix / norm
        # Return the normalized matrix
        return normalized
    
    # A function that returns the probabilities of each letter following the given letters
    # This can be used within this class by the choose_next_letter function
    # Or it can be used outside this class to get the probabilities of each letter following a given string
    # Using it outside this class allows samplers to manipulate the probabilities separately
    # Turns out, using raw probabilities continuously is a bad idea
    def get_probabilities(self,letters):
        # If there are too many letters, trim it down to the last n-1 letters
        if len(letters) > self.n - 1:
            letters = letters[-(self.n - 1):]
        # If there are too few letters, pad it with spaces at the start
        elif len(letters) < self.n - 1:
            letters = ' ' * (self.n - 1 - len(letters)) + letters
        # Convert the letters to numbers
        nums = [self.letter_to_num(letter) for letter in letters]
        # Get the row of the normalized matrix corresponding to the given letters
        row = self.normalizedmatrix[tuple(nums)]
        # Return the row
        return row
    
    # A function that randomly chooses the next letter based on the normalized matrix
    # This is called by the generate_word function to keep getting new letters
    def choose_next_letter(self,letters):
        # Get the row of the normalized matrix corresponding to the given letters
        row = self.get_probabilities(letters)
        # Choose a random number between 0 and 1
        r = random.random()
        # Initialize the cumulative probability
        cumprob = 0
        # For each letter in the alphabet
        for i in range(self.lenalpha):
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
            # If no start string is given, start with n-1 spaces
            start = ' ' * (self.n - 1)
        # Create a variable for the word
        word = start
        nextletter = None
        # While the next letter isn't a space
        while nextletter != ' ':
            # Get the last n-1 letters of the word
            lastletters = word[-(self.n - 1):]
            # Get the next letter
            nextletter = self.choose_next_letter(lastletters)
            # Add the next letter to the word
            word += nextletter
        # Strip out the leading and trailing spaces
        word = word.strip()
        # Return the word
        return word

# So, here start the sampler classes
# These manipulate the probabilities of the letters
# They're used because raw next token probabilities are a bad idea
# They're also used because I want to see if I can do it
# At some point, I'll create a base sampler class
# And then have each sampler inherit from that
# But for now, I'm just going to make them all separately
# And figure out the generalizations later

# This class attempts to make words that are of normal length
# It does this by manipulating the probabilities based on the length of the word so far
class WordLenSampler:
    def __init__(self,wordlist):
        # Suck in the word list
        self.wordlist = wordlist
        # Get the word length probabilities
        self.wordlengthprobs = self.get_word_length_probabilities()

    # Turns the list of words into a list of word lengths
    def get_word_lengths(self):
        return [len(word) for word in self.wordlist]
    
    # Turns the list of word lengths into a list of probabilities
    # The probability of a word length is the number of words of that length divided by the total number of words
    def get_word_length_probabilities(self):
        # Get the word lengths
        wordlengths = self.get_word_lengths()
        # Get the number of words of each length
        wordlengthcounts = [wordlengths.count(i) for i in range(1,max(wordlengths)+1)]
        # Get the total number of words
        totalwords = len(wordlengths)
        # Get the probabilities
        wordlengthprobs = [wordlengthcount / totalwords for wordlengthcount in wordlengthcounts]
        # Return the probabilities
        return wordlengthprobs
        



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
testmatrix = NgramMatrix(alphabet,3)
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
# So, I only had to change one line down below
# And it all still works
# ...I'm going to be honest, I'm a little surprised
# And at some point I have to try 3-grams, 4-grams, etc.


















# Load the word list from the file
wordlist = get_word_list('wordlist.txt')

# A quick test to make sure the functions work
#print(wordlist[0:10])
#print(testmatrix.letter_to_num('a'))
#print(testmatrix.num_to_letter(1))
#print(testmatrix.word_to_num_list('hello'))
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
#testmatrix.print_matrix(testmatrix.cumulativematrix)

# How about printing the normalized matrix?
#testmatrix.print_matrix(testmatrix.normalizedmatrix)

# How about exporting the matrix to a csv file?
# testmatrix.export_matrix(testmatrix.cumulativematrix,'testmatrix.csv')

# Now what about the normalized matrix?
# testmatrix.export_matrix(testmatrix.normalizedmatrix,'testmatrix_normalized.csv')

# How about the ten most common 2-grams?
#print(testmatrix.most_common(testmatrix.cumulativematrix,10))

# How about the hundred least common 2-grams?
#print(testmatrix.least_common(testmatrix.cumulativematrix,100))

# Now, let's generate 10 random words
for i in range(10):
    print(testmatrix.generate_word())

# Okay, here, test the sampler
testlensampler = WordLenSampler(wordlist)
print(testlensampler.wordlengthprobs)
import random
import time
import numpy as np

# This class represents an n-gram tensor
# It's an n-dimensional array of numbers
# The first index is the first letter of the n-gram
# And so on and so forth
class Ngrams:
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
        self.basetensor = np.ones([self.lenalpha for i in range(self.n)],dtype=int)
        # Create the cumulative matrix, which is the matrix we're actually going to use
        # It starts out as a copy of the base matrix
        # As we train the matrix, we'll update the cumulative matrix
        self.cumulativetensor = self.basetensor.copy()
        # The normalized matrix is the cumulative matrix, but with each row normalized
        # This is used to generate random words
        self.normalizedtensor = self.normalize_tensor(self.cumulativetensor)
        # A dictionary that can hold arbitrary matrices
        # This is used to keep track of matrices that have been trained on different word lists
        # The key is the name of the matrix
        # The value is the matrix itself
        self.tensors = {'cumulative':self.cumulativetensor,'normalized':self.normalizedtensor}
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
    
    # Adds a tensor to the dictionary of tensors
    # The key is the name of the tensor
    # By default, the tensor is the base tensor
    def add_tensor(self,name,matrix=None):
        # If no matrix is given, use the base matrix
        if matrix == None:
            matrix = self.basematrix
        # Add a copy of the matrix to the dictionary
        self.matrices[name] = matrix.copy()
        # Return the matrix
        return matrix
    
    # Takes an in-progress tensor and a word and updates the tensor
    def update_matrix(self,word,tensor=None):
        # If no tensor is given, use the cumulative tensor
        if tensor is None:
            tensor = self.cumulativetensor
        # Get the n-grams from the word
        ngrams = self.word_to_ngrams(word)
        # For each n-gram in the word
        for ngram in ngrams:
            # If the matrix being updated is not the cumulative matrix
            # Then you update the named matrix
            if tensor is not self.cumulativetensor:           
                tensor[ngram] += 1
            # You ALWAYS update the cumulative matrix
            self.cumulativetensor[ngram] += 1

    # Takes in a list of words and trains the tensor on those words
    def train_tensor(self,wordlist,tensor=None):
        # If no matrix is given, use the cumulative matrix
        if tensor is None:
            tensor = self.cumulativetensor
        # For each word in the word list
        for word in wordlist:
            # Update the cumulative matrix
            self.update_matrix(word,tensor)
        # Update the normalized matrix
        self.normalizedtensor = self.normalize_tensor(self.cumulativetensor)
        # Return the cumulative matrix
        return self.cumulativetensor
    
    # Since the tensor is an n-dimensional array, it's not easy to print
    # So I'm not going to bother with that for now
    def print_matrix(self,matrix):
        pass

    # Not going to bother with this one either
    def export_matrix(self,matrix,filename):
        pass

    # A function that "flattens" the tensor into a list of tuples
    # The first element of the tuple is the n-gram (as a string of letters, NOT NUMBERS)
    # The second element of the tuple is the frequency of that n-gram
    def flatten_matrix(self,tensor):
        # Initialize the flat list
        flat = []
        # Enumerate the tensor
        for index,value in np.ndenumerate(tensor):
            ngram = self.num_list_to_word(index)
            # Add the index and value to the flat list
            flat.append((ngram,value))
        # Return the flat list
        return flat
        
    # A function that returns the n most common n-grams in the tensor
    # The return value is a list of tuples
    # The first element of the tuple is the n-gram (as a string of letters, NOT NUMBERS)
    # The second element of the tuple is the frequency of that n-gram
    def most_common(self,tensor,n):
        # Flatten the matrix
        flat = self.flatten_matrix(tensor)
        # Sort the flattened matrix by frequency
        flat = sorted(flat,key=lambda x: x[1],reverse=True)
        # Return the n most common 2-grams
        return flat[0:n]
    
    # A function that returns the n least common n-grams in the tensor
    # The return value is a list of tuples
    # The first element of the tuple is the n-gram (as a string of letters, NOT NUMBERS)
    # The second element of the tuple is the frequency of that n-gram
    def least_common(self,tensor,n):
        # Flatten the matrix
        flat = self.flatten_matrix(tensor)
        # Sort the flattened matrix by frequency
        flat = sorted(flat,key=lambda x: x[1])
        # Return the n least common 2-grams
        return flat[0:n]
    
    # A function that normalizes a matrix
    # Each row is normalized so that the sum of the row is 1
    # These end up being the weights used to generate random words
    def normalize_tensor(self,tensor):
        # Normalize the array along the last axis
        # This is the axis that corresponds to the last letter of the n-gram
        # This is the axis that we want to sum over
        sum = np.sum(tensor,axis=-1,keepdims=True)
        # Divide the matrix by the sum
        normalized = tensor / sum
        # Return the normalized tensor
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
        row = self.normalizedtensor[tuple(nums)]
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
    # The word is generated using the normalized tensor
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
        # Get the cumulative word length probabilities
        self.cumwordlengthprobs = self.get_cumulative_word_length_probabilities()

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
    
    # Takes the probabilities by word length and turns it into cumulative probabilities
    def get_cumulative_word_length_probabilities(self):
        # Get the word length probabilities
        wordlengthprobs = self.wordlengthprobs
        # Initialize the cumulative probabilities
        cumprobs = [0 for i in range(len(wordlengthprobs))]
        # For each word length
        for i in range(len(wordlengthprobs)):
            # Add the probability of the word length to the cumulative probability
            cumprobs[i] = cumprobs[i-1] + wordlengthprobs[i]
        # Return the cumulative probabilities
        return cumprobs
    
    # Determines the likelihood that the word should end at its current length
    # Uses the cumulative probabilities
    def get_end_word_probability(self,word):
        length = len(word)
        # If the word is too long, return 1
        if length > len(self.cumwordlengthprobs):
            return 1
        # Otherwise, return the cumulative probability of the word length
        return self.cumwordlengthprobs[length]

    # Takes in a row of probabilities and returns a new row of probabilities
    # The idea: roll a die to determine if the word should end
    # If the word should end, then the probability of the space token is increased to 1
    # And all other probabilities are set to 0
    # Otherwise, the probabilities are unchanged
    def get_new_row(self,row,word):
        # Get the probability that the word should end
        endwordprob = self.get_end_word_probability(word)
        # Roll a die
        r = random.random()
        # If the word should end
        if r < endwordprob:
            # Initialize the new row
            newrow = [0 for i in range(len(row))]
            # Set the space token to 1
            newrow[0] = 1
            # Return the new row
            return newrow
        # If the word shouldn't end
        else:
            # Return the original row
            return row

# This class throws out a certain number of low probability tokens
# It keeps the top k tokens
class TopKSampler:
    def __init__(self):
        pass

    # Get the row of the normalized matrix corresponding to the given letters
    def get_probs(self,letters,ngramtensor):
        # Get the row of the normalized matrix corresponding to the given letters
        row = ngramtensor.get_probabilities(letters)
        # Return the row
        return row

    # Takes in a row of probabilities and returns a list of the top k tokens
    def get_top_k(self,row,k):
        # Convert the row to a list of tuples
        # The first element of the tuple is the index
        # The second element of the tuple is the probability
        row = [(i,row[i]) for i in range(len(row))]
        # Sort the row by probability
        row = sorted(row,key=lambda x: x[1],reverse=True)
        # Get the top k tokens
        topklist = row[0:k]
        # Return the top k tokens
        return topklist
    
    # Takes the top k tokens and creates a new row of probabilities
    # The top k tokens keep their original probabilities for now
    # The rest of the tokens are given 0 probability
    def get_new_row(self,topklist,row):
        # Initialize the new row
        newrow = [0 for i in range(len(row))]
        # For each token in the top k
        for token in topklist:
            # Add the token and its probability to the new row
            newrow[token[0]] = token[1]
        # Return the new row
        return newrow
    
    # Normalizes the row of probabilities
    # Normalization here means that the sum of the row is 1
    def normalize_row(self,row):
        # Normalize the row
        norm = np.sum(row)
        normalized = row / norm
        # Return the normalized row
        return normalized
    
    # Put it all together to get the new probabilities
    # In goes the current string of tokens, the current tensor, and the top k value
    # Out comes the new probabilities
    def get_new_probs(self,letters,ngramtensor,k):
        # Get the row of probabilities
        baseprobs = self.get_probs(letters,ngramtensor)
        # Get the top k token and put them in a row
        topklist = self.get_top_k(baseprobs,k)
        topkrow = self.get_new_row(topklist,baseprobs)
        # Normalize the row
        newprobs = self.normalize_row(topkrow)
        # Return the new probabilities
        return newprobs
    
# This class keeps new tokens that add up to a certain percentage of the total probability
# The rest of the tokens are thrown out
class TopPSampler:
    def __init__(self):
        pass

    # Get the row of the normalized matrix corresponding to the given letters
    def get_probs(self,letters,ngrammatrix):
        # Get the row of the normalized matrix corresponding to the given letters
        row = ngrammatrix.get_probabilities(letters)
        # Return the row
        return row
    
    # Takes in a row of probabilities and returns the tokens that add up to the top p percent
    def get_top_p(self,row,p):
        # Convert the row to a list of tuples
        # The first element of the tuple is the index
        # The second element of the tuple is the probability
        row = [(i,row[i]) for i in range(len(row))]
        # Sort the row by probability
        row = sorted(row,key=lambda x: x[1],reverse=True)
        # Initialize the cumulative probability
        cumprob = 0
        # Initialize the top p list
        topp = []
        # For each token in the row
        for token in row:
            # Add the probability of the token to the cumulative probability
            cumprob += token[1]
            # Add the token to the top p list
            topp.append(token)
            # If the cumulative probability is greater than the top p percent
            if cumprob > p:
                # Return the top p list
                return topp
        # If we get here, something went wrong
        # I don't know what, but something
        # So raise an error
        raise Exception('Something went wrong in get_top_p')
    
    # Takes the top p tokens and creates a new row of probabilities
    # The top p tokens keep their original probabilities for now
    # The rest of the tokens are given 0 probability
    def get_new_row(self,topplist,row):
        # Initialize the new row
        newrow = [0 for i in range(len(row))]
        # For each token in the top p
        for token in topplist:
            # Add the token and its probability to the new row
            newrow[token[0]] = token[1]
        # Return the new row
        return newrow
    
    # Normalizes the row of probabilities
    # Normalization here means that the sum of the row is 1
    def normalize_row(self,row):
        # Normalize the row
        norm = np.sum(row)
        normalized = row / norm
        # Return the normalized row
        return normalized
    
    # Put it all together to get the new probabilities
    # In goes the current string of tokens, the current tensor, and the top p value
    # Out comes the new probabilities
    def get_new_probs(self,letters,ngramtensor,p):
        # Get the row of probabilities
        baseprobs = self.get_probs(letters,ngramtensor)
        # Get the top p token and put them in a row
        topplist = self.get_top_p(baseprobs,p)
        topprow = self.get_new_row(topplist,baseprobs)
        # Normalize the row
        newprobs = self.normalize_row(topprow)
        # Return the new probabilities
        return newprobs

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
testtensor = Ngrams(alphabet,3)
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
print(testtensor.word_to_ngrams('hello'))

# Some words to train the matrix on
# words = ["hello","world"]
# Commented that out because now we're going to use the whole word list
words = wordlist

starttime = time.time()

# Train the matrix on the words
testtensor.train_tensor(words)

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
    print(testtensor.generate_word())

# Okay, here, test the sampler
testlensampler = WordLenSampler(wordlist)
print(testlensampler.wordlengthprobs)

# Okay, here, test the top k sampler
testtopksampler = TopKSampler()
letters = 'hello'
print(testtopksampler.get_probs(letters,testtensor))
print(testtopksampler.get_new_probs(letters,testtensor,5))
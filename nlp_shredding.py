#! /usr/bin/env python
# ---------
# nlp_op.py
# Nicholas Shelly (nshelly@stanford.edu)

"""
(1) Builds list of character n-grams, where n is the size of the shredded columns
    In the case of the AI class NLP, these are character bigrams.
 
(2) Calculates probability table.  For each column find the probability of the 
    column coming after or before the other columns -- take the product of 
    probabilities over the column. 
    For example, given bigram of "de", what is the probability of the next 
    being "nf"?
    P (wn | wn-1) eg ("nf"|"de")
    C(wn-1 wn) / C(wn)

(3) Sorts each column's neighbors for before and after so the most likely 
    connection is earliest.

(3) Starting with the strongest connection, the greedy algorithm
    builds onto the sequence either to the left or right based on the next 
    most likely bigram connection, until all columns are marked.  

(4) Prints the sequence.
"""

import re
from operator import itemgetter

DEBUG = 1
#CORPUS = "usaa_notags.txt"      #  Source: http://www.comp.leeds.ac.uk/eric/db32/us/
CORPUS = "wp_complete.txt"      #  Source: http://www.comp.leeds.ac.uk/eric/db32/us/
CORPUS_CHARS = 10000000
CIPHER = """
|de|  | f|Cl|nf|ed|au| i|ti|  |ma|ha|or|nn|ou| S|on|nd|on|
|ry|  |is|th|is| b|eo|as|  |  |f |wh| o|ic| t|, |  |he|h |
|ab|  |la|pr|od|ge|ob| m|an|  |s |is|el|ti|ng|il|d |ua|c |
|he|  |ea|of|ho| m| t|et|ha|  | t|od|ds|e |ki| c|t |ng|br|
|wo|m,|to|yo|hi|ve|u | t|ob|  |pr|d |s |us| s|ul|le|ol|e |
| t|ca| t|wi| M|d |th|"A|ma|l |he| p|at|ap|it|he|ti|le|er|
|ry|d |un|Th|" |io|eo|n,|is|  |bl|f |pu|Co|ic| o|he|at|mm|
|hi|  |  |in|  |  | t|  |  |  |  |ye|  |ar|  |s |  |  |. |"""
SPACE= "_"
N = 2        # Number of characters in slice
ALPHA = 1.0  # For Laplace Smoothing

class ShreddedMessage: 

    def __init__(self, CIPHER):
        with open(CORPUS, 'r') as f:
            corpus = f.read()
            corpus = re.sub('[^a-z\n|]', SPACE, corpus.lower())
        words = corpus[:CORPUS_CHARS]
        if DEBUG:
            print "Building bigrams for corpus (%d characters)..." % len(corpus)
        
        self.ngrams = self.find_ngrams(words)
        ciphertext = re.sub('[^a-z\n|]', SPACE, CIPHER.lower()).strip()
        if DEBUG: print "Ciphertext:\n%s" % ciphertext
        self.cipher = [line.split('|')[1:-1] for line in \
                        ciphertext.split('\n')]
        self.min_value = float(1) / CORPUS_CHARS

    def find_ngrams(self, words):
        """ Tokenize words """
        ngrams = {}
        for i in range(len(words) - N):
            pair1 = words[i:i+N]
            # Second character of first bigram is space, so consider bigram as space
            if pair1[N-1] == SPACE:
                pair1 = SPACE*2
            pair2 = words[i+N:i+N+N]
            # If following bigram begins space, just fill both with space
            if pair2[0] == SPACE:
                pair2 = SPACE*2
            if (pair1, pair2) == (SPACE*2, SPACE*2):
                continue
            if pair1 not in ngrams:
                ngrams[pair1] = []
            ngrams[pair1].append(pair2)
        return ngrams

    def Pngram(self, curr, prev):
        """ 
        Returns the probability of curr ngram, given prev ngram. 
        e.g. P(prev+curr|prev)
        """
        # Penalize for spaces to give more influence to building words
        if curr == SPACE*2 or prev == SPACE*2:
            return self.min_value

        if prev[-1] == SPACE:
            prev = SPACE*2
        if curr[0] == SPACE:
            curr = SPACE*2

        # Could implement Laplace Smoothing here:
        # (Count(Word) + ALPHA) / (Total + all_bigrams * 1)
        if prev in self.ngrams:
            Cboth = self.ngrams[prev].count(curr)
            Cprev = len(self.ngrams[prev])
            if DEBUG > 1:
                print "C('%s %s') = %d / C('%s') = %d => %f" % \
                        (prev, curr, Cboth, prev, Cprev, float(Cboth)/Cprev)
            return float(Cboth) / Cprev or self.min_value
        else:
            return self.min_value
        
    def print_pair(self, path, p):
        print "best path: col %d to col %d - P(%e)" % (path[0], path[1], p)
        for row in self.cipher:
            print "|".join("%s" % row[p] for p in path)

    def print_sequence(self, seq):
        for row in self.cipher:
            print "|".join("%s" % row[s] for s in seq)

    def solve(self):
        cols = len(self.cipher[0])
        rows = len(self.cipher)
        # Table of probabilities that column j comes before i
        before = [[] for c in range(cols)]  
        # Table of probabilities that column j comes after i
        after = [[] for c in range(cols)]   
        print "Calculating probability table..."
        for i in range(cols):
            for j in range(cols):
                before_prob = 1       # Likelihood jth column precedes ith
                after_prob = 1        # Likelihood jth column follows ith
                if i != j:
                    if DEBUG > 1: print "Checking col %d" % j 
                    for r in range(rows):
                        curr = self.cipher[r][i]
                        check = self.cipher[r][j]
                        # Probabiliy of j coming after current col
                        after_prob *= self.Pngram(check, curr)
                        # Probabiliy of j coming before current col
                        before_prob *= self.Pngram(curr, check)
                    before[i].append((j, before_prob)) 
                    after[i].append((j, after_prob))

            after[i].sort(key=itemgetter(1), reverse=True)
            before[i].sort(key=itemgetter(1), reverse=True)

        marked = [0 for c in range(cols)]
        left = max((p[0] for p in before), key=itemgetter(1))[0]
        right = after[left][0][0]
        sequence = [left, right]
        marked[left] = 1 
        marked[right] = 1 

        # Continue to pick the best available neighbor and append to sequence 
        while len(sequence) < cols: 
            if DEBUG: self.print_sequence(sequence) 
            i = 0
            while i < cols-1 and marked[before[left][i][0]]:
                i += 1
            next_left = before[left][i]
            i = 0
            while i < cols-1 and marked[after[right][i][0]]:
                i += 1
            next_right = after[right][i]
            if next_left[1] > next_right[1]:
                left = next_left[0]
                marked[left] = 1
                sequence.insert(0, left)
                if DEBUG: print "Adding col %d to left" % left
                if DEBUG > 1: self.print_pair(sequence[0:2], next_left[1]) 
            else:
                right = next_right[0]
                marked[right] = 1
                sequence.append(right)
                if DEBUG: print "Adding col %d to right" % right
                if DEBUG > 1: self.print_pair(sequence[-2:], next_left[1]) 
            if DEBUG > 2: raw_input('ok?')
        self.print_sequence(sequence) 

def main():
    message = ShreddedMessage(CIPHER)
    message.solve()
            
if __name__ == '__main__':
    main()

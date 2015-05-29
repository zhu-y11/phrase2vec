#-*- coding: utf-8 -*-
'''
Training example class
@author: lpeng
'''

class Instance(object):
    '''A reordering training example'''
  
    def __init__(self, words, freq, idx):
        '''
        Args:
        words: numpy.array (an int array of word indices)
        freq: frequency of this training example
        '''
        self.words = words
        self.freq = freq
        self.idx = idx

    def __str__(self):
        parts = []
        parts.append(' '.join([str(i) for i in self.words]))
        parts.append(str(self.freq))
        parts.append(' '.join( idx ) )
        return ' ||| '.join(parts)
    
    @classmethod
    def parse_from_str(cls, line, word_vector):
        '''The format of the line should be like the following:
        src_word1, src_word2,..., src_wordn ||| freq || id
        freq is optional
        ''' 
        pos0 = line.find(' ||| ')
        pos1 = line.find( ' ||| ', pos0 + 5 )
        words = [word_vector.get_word_index(word) for word in line[0:pos0].split()]
        freq = int( line[ pos0 + 5: pos1 ] )
        idx = line[pos1 + 5:].strip().split( ' ' )
        print line, freq, idx
        if( line.find( '$X' ) >= 0 ):
            assert( len( idx ) == freq )

        return Instance( words, freq, idx )

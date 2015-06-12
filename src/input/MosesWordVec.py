import random as rd
#from gensim.models import Word2Vec as w2v

infile = open( 'phrase-table_30k', 'r' )
outfile_svec = open( 'moses_source_word_vector', 'w' )
outfile_tvec = open( 'moses_target_word_vector', 'w' )

src_word_vec = []
trg_word_vec = []

for line in infile:
    line = line.decode( 'utf-8' )
    linevec = line.strip().split( ' ||| ' )
    src = linevec[0]
    trg = linevec[1]
    
    srcvec = src.strip().split( ' ' )
    trgvec = trg.strip().split( ' ' )
    src_word_vec.extend( srcvec )
    trg_word_vec.extend( trgvec )
    
infile.close()

src_word_vec = list( set( src_word_vec ) )
trg_word_vec = list( set( trg_word_vec ) )

#srcmodel = w2v.load('cnmodel')
#trgmodel = w2v.load('enmodel')
for src_word in src_word_vec:
    outfile_svec.write( src_word.encode( 'utf-8' ) + ' ')
    for i in xrange( 49 ):
        outfile_svec.write( str( rd.gauss( 0, 1 ) ) + ' ' )
    outfile_svec.write( str( rd.gauss( 0, 1 ) ) + '\n' )
outfile_svec.close()

for trg_word in trg_word_vec:
    outfile_tvec.write( trg_word.encode( 'utf-8' ) + ' ')
    for i in xrange( 49 ):
        outfile_tvec.write( str( rd.gauss( 0, 1 ) ) + ' ' )
    outfile_tvec.write( str( rd.gauss( 0, 1 ) ) + '\n' )
outfile_tvec.close()

'''
for src_word in src_word_vec:
    outfile_svec.write( src_word.encode( 'utf-8' ) + ' ')
    if src_word in srcmodel:
        tplist = srcmodel[src_word]
        for i in xrange( 3 ):
            outfile_svec.write( str( tplist[i] ) + ' ' )
        outfile_svec.write( str( tplist[3] ) + '\n' )
    else:
        for i in xrange( 3 ):
            outfile_svec.write( str( rd.gauss( 0, 1 ) ) + ' ' )
        outfile_svec.write( str( rd.gauss( 0, 1 ) ) + '\n' )
outfile_svec.close()

for trg_word in trg_word_vec:
    outfile_tvec.write( trg_word.encode( 'utf-8' ) + ' ')
    if trg_word in trgmodel:
        tplist = trgmodel[trg_word]
        for i in xrange( 3 ):
            outfile_tvec.write( str( tplist[i] ) + ' ' )
        outfile_tvec.write( str( tplist[3] ) + '\n' )
    else:
        for i in xrange( 3 ):
            outfile_tvec.write( str( rd.gauss( 0, 1 ) ) + ' ' )
        outfile_tvec.write( str( rd.gauss( 0, 1 ) ) + '\n' )
outfile_tvec.close()
'''

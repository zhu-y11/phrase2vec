import random as rd
from gensim.models import Word2Vec as w2v

infile = open( 'ruleTable.txt', 'r' )
outfile_source = open( 'source_ruleTable', 'w' )
outfile_target = open( 'target_ruleTable', 'w' )
outfile_svec = open( 'source_word_vector', 'w' )
outfile_tvec = open( 'target_word_vector', 'w' )

src_word_vec = []
trg_word_vec = []

for line in infile:
    line = line.decode( 'utf-8' )
    linevec = line.strip().split( ' ||| ' )
    src = linevec[0]
    trg = linevec[1]
    score = linevec[2]
    scorevec = score.strip().split( ' ' )
    freq = scorevec[2]
    scorevec = scorevec[3:]
    
    srcvec = src.strip().split( ' ' )
    trgvec = trg.strip().split( ' ' )
    src_word_vec.extend( srcvec )
    trg_word_vec.extend( trgvec )
    
    outfile_source.write( src.encode( 'utf-8' ) + ' ||| ' + freq.encode( 'utf-8' ) + ' ||| '  )
    outfile_target.write( trg.encode( 'utf-8' ) + ' ||| ' + freq.encode( 'utf-8' ) + ' ||| '  )
    for id in scorevec:
        outfile_source.write( id + ' ' )
        outfile_target.write( id + ' ' )
    outfile_source.write( '\n' )
    outfile_target.write( '\n' )

infile.close()
'''
infile = open( 'ruleFreq_1.txt', 'r' )
for line in infile:
    line = line.decode( 'utf-8' )
    linevec = line.strip().split( ' ||| ' )
    src = linevec[0]
    trg = linevec[1]
    score = linevec[2]
    scorevec = map( int, score.strip().split( ' ' )[3:] )
    idx = scorevec[0] 
 
    if src.count( ' ' ) < 1 or trg.count( ' ' ) < 1:
        continue

    if src.find( '$X' ) < 0:
        outfile_source.write( src.encode( 'utf-8' ) + ' ||| ' + '1' + ' ||| '  )
        outfile_target.write( trg.encode( 'utf-8' ) + ' ||| ' + '1' + ' ||| '  )
        for id_ele in hiero_map.keys():
            if idx in hiero_map[id_ele]:
                outfile_source.write( str( id_ele ) + ' ' )
                outfile_target.write( str( id_ele ) + ' ' )
                
        outfile_source.write( '\n' )
        outfile_target.write( '\n' )
infile.close()
'''
outfile_source.close()
outfile_target.close()

src_word_vec = list( set( src_word_vec ) )
trg_word_vec = list( set( trg_word_vec ) )

#src_word_vec.remove( '$X_1' )
#src_word_vec.remove( '$X_2' )
#trg_word_vec.remove( '$X_1' )
#trg_word_vec.remove( '$X_2' )

srcmodel = w2v.load('cnmodel')
trgmodel = w2v.load('enmodel')
for src_word in src_word_vec:
    outfile_svec.write( src_word.encode( 'utf-8' ) + ' ')
    if src_word in srcmodel:
        tplist = srcmodel[src_word]
        for i in xrange( 49 ):
            outfile_svec.write( str( tplist[i] ) + ' ' )
        outfile_svec.write( str( tplist[49] ) + '\n' )
    else:
        for i in xrange( 49 ):
            outfile_svec.write( str( rd.gauss( 0, 1 ) ) + ' ' )
        outfile_svec.write( str( rd.gauss( 0, 1 ) ) + '\n' )
outfile_svec.close()

for trg_word in trg_word_vec:
    outfile_tvec.write( trg_word.encode( 'utf-8' ) + ' ')
    if trg_word in trgmodel:
        tplist = trgmodel[trg_word]
        for i in xrange( 49 ):
            outfile_tvec.write( str( tplist[i] ) + ' ' )
        outfile_tvec.write( str( tplist[49] ) + '\n' )
    else:
        for i in xrange( 49 ):
            outfile_tvec.write( str( rd.gauss( 0, 1 ) ) + ' ' )
        outfile_tvec.write( str( rd.gauss( 0, 1 ) ) + '\n' )
outfile_tvec.close()

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument( '-score_file', required = True, help = 'The file containing the similarity' )
    parser.add_argument( '-filtered_file', required = True, help = 'The filtered file by threshold' )
    parser.add_argument( '-threshold', type = float, required = True, help = 'The threshold for filtering')
    options = parser.parse_args()
    
    score_file = open( options.score_file, 'r' )
    filtered_file = open( options.filtered_file, 'w' )
    threshold = options.threshold
    print 'start filtering...'
    for line in score_file:
        linevec = line.strip().split( ' ||| ' )
        src_word = linevec[0]
        trg_word = linevec[1]
        scorevec = map( float, linevec[2].strip().split( ',' ) )
        score1 = scorevec[0]
        score2 = scorevec[1]
        if score1 < threshold or score2 < threshold:
            continue
        filtered_file.write( ' ||| '.join( linevec ) )
        filtered_file.write( '\n' )

    score_file.close()
    filtered_file.close()
    print 'Filtering Done.'


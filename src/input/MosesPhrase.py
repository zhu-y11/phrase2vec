import random as rd

infile = open( 'phrase-table_30k', 'r' )
outfile_source = open( 'moses_source_ruleTable', 'w' )
outfile_target = open( 'moses_target_ruleTable', 'w' )
outfile_data = open( 'moses_data', 'w' )

for line in infile:
    line = line.decode( 'utf-8' )
    linevec = line.strip().split( ' ||| ' )
    src = linevec[0]
    trg = linevec[1]
    data = ' ||| '.join( linevec[2:] )
    
    srcvec = src.strip().split( ' ' )
    trgvec = trg.strip().split( ' ' )
    
    outfile_source.write( src.encode( 'utf-8' )  )
    outfile_target.write( trg.encode( 'utf-8' )  )
    outfile_data.write( data.encode( 'utf-8' ) )
    outfile_source.write( '\n' )
    outfile_target.write( '\n' )
    outfile_data.write( '\n' )

infile.close()
outfile_source.close()
outfile_target.close()
outfile_data.close()

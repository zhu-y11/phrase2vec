phrase_map = {}
zero_id = 0

infile = open('ruleFreq_1.txt','r' )
outfile = open('ruleTable.txt','w' )

for line in infile:
    if '$X' in line:
        continue

    linevec = line.strip().split( ' ||| ' )
    scorevec = linevec[2].strip().split( ' ' )
    p0 = scorevec[0]
    p1 = scorevec[1]
    p2 = scorevec[2]
    scorevec = scorevec[3:]
    for score in scorevec:
        phrase_map[score] = zero_id
    outfile.write( linevec[0] + ' ||| ' + linevec[1] + ' ||| ' + 
                p0 + ' ' + p1 + ' ' + p2 + ' ' + str( zero_id ) + '\n' )
    zero_id += 1

infile.close()

infile = open('ruleFreq_1.txt','r' )
for line in infile:
    if '$X_1' in line and not '$X_2' in line :
        linevec = line.strip().split( ' ||| ' )
        scorevec = linevec[2].strip().split( ' ' )
        p0 = scorevec[0]
        p1 = scorevec[1]
        p2 = scorevec[2]
        scorevec = scorevec[3:]
        outfile.write( linevec[0] + ' ||| ' + linevec[1] + ' ||| ' + p0 + ' ' + p1 + ' ' + p2 + ' ' )
        for score in scorevec:
            outfile.write( str(phrase_map[score]) + ' ' )
        outfile.write( '\n' )
    elif '$X_2' in line:
        linevec = line.strip().split( ' ||| ' )
        scorevec = linevec[2].strip().split( ' ' )
        p0 = scorevec[0]
        p1 = scorevec[1]
        p2 = scorevec[2]
        scorevec = scorevec[3:]
        outfile.write( linevec[0] + ' ||| ' + linevec[1] + ' ||| ' + p0 + ' ' + p1 + ' ' + p2 + ' ' )
        for score in scorevec:
            subscorevec = score.strip().split( ',' )
            outfile.write( str(phrase_map[subscorevec[0]]) + ',' + str(phrase_map[subscorevec[1]]) + ' ' )
        outfile.write( '\n' )
    else:
        continue
infile.close()
outfile.close()


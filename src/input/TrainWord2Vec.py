from gensim.models import Word2Vec as w2v

class MySentences(object):
    def __init__(self, filename):
        self.filename = filename
 
    def __iter__(self):
        for line in open( self.filename ):
            yield line.split()

if __name__ == '__main__':
    srcfile = MySentences('/global-mt/lpeng/academic/neural-moses/corpus/bilingual/1227K-lowercase/3-clean/1227K-lowercase.chi-eng.tok.norm.clean.chi')
    cnmodel = w2v(srcfile, workers=4, size=4)
    cnmodel.save('/data/disk1/private/zy/phrase_str2vec/src/input/cnmodel')
    trgfile = MySentences('/global-mt/lpeng/academic/neural-moses/corpus/bilingual/1227K-lowercase/3-clean/1227K-lowercase.chi-eng.tok.norm.clean.eng')
    enmodel = w2v(trgfile, workers=4, size=4)
    enmodel.save('/data/disk1/private/zy/phrase_str2vec/src/input/enmodel')
    


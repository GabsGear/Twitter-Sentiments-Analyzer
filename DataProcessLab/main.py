#disable=W0614
import TwitterRT
import pandas as pd
import scienceLAB

def main():
    #Routine for analysis using NLTK
    source = TwitterRT.AnalyzerTwiterRT()
    tweets = source.getCSV('Trump', 100)
    lab = scienceLAB.Lab()
    tweets = pd.read_csv(
        '/home/gabs/Backend/Backend/DataProcessLab/sentiments /inputs/result.csv', encoding="ISO-8859-1")

    tweets = lab.cleanData(tweets)
    result = lab.sentimentAnalysis(tweets)
    print('nltk result')
    print(result)

    #FOR use textblob
    analysis = source.analysis('Trump', 100)
    print('Sentiments Average, -1 Bad, +1 Good = ' + str(analysis))

main()
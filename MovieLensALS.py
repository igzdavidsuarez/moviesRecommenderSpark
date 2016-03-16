#!/usr/bin/env python

import sys
import itertools
from math import sqrt
from operator import add
from os.path import join, isfile, dirname

from pyspark import SparkConf, SparkContext
from pyspark.mllib.recommendation import ALS


def parseRating(line):
    """
    Parses a rating record in MovieLens format userId::movieId::rating::timestamp .
    """
    fields = line.strip().split("::")
    return long(fields[3]) % 10, (int(fields[0]), int(fields[1]), float(fields[2]))

def parseMovie(line):
    """
    Parses a movie record in MovieLens format movieId::movieTitle .
    """
    fields = line.strip().split("::")
    return int(fields[0]), fields[1]

def loadRatings(ratingsFile):
    """
    Load ratings from file.
    """
    if not isfile(ratingsFile):
        print "File %s does not exist." % ratingsFile
        sys.exit(1)
    f = open(ratingsFile, 'r')
    ratings = filter(lambda r: r[2] > 0, [parseRating(line)[1] for line in f])
    f.close()
    if not ratings:
        print "No ratings provided."
        sys.exit(1)
    else:
        return ratings

def computeRmse(model, data, n):
    """
    Compute RMSE (Root Mean Squared Error).
    """
    predictions = model.predictAll(data.map(lambda x: (x[0], x[1])))
    predictionsAndRatings = predictions.map(lambda x: ((x[0], x[1]), x[2])) \
      .join(data.map(lambda x: ((x[0], x[1]), x[2]))) \
      .values()
    return sqrt(predictionsAndRatings.map(lambda x: (x[0] - x[1]) ** 2).reduce(add) / float(n))

# /home/davidsuarez/Documents/MACHINE_LEARNING/spark-1.6.1-bin-hadoop2.6/python/pythonMovies/data/movielens/medium/movies.dat
# /home/davidsuarez/Documents/MACHINE_LEARNING/spark-1.6.1-bin-hadoop2.6/python/pythonMovies/data/movielens/personalRatings.txt

if __name__ == "__main__":
    if (len(sys.argv) != 3):
        print "Usage: /path/to/spark/bin/spark-submit --driver-memory 2g " + \
          "MovieLensALS.py movieLensDataDir personalRatingsFile"
        sys.exit(1)

    # set up environment
    conf = SparkConf() \
      .setAppName("MovieLensALS") \
      .set("spark.executor.memory", "2g")
    sc = SparkContext(conf=conf)

    # load personal ratings
    myRatings = loadRatings(sys.argv[2])
    myRatingsRDD = sc.parallelize(myRatings, 1)
    
    # load ratings and movie titles

    movieLensHomeDir = sys.argv[1]

    # ratings is an RDD of (last digit of timestamp, (userId, movieId, rating))
    ratings = sc.textFile(join(movieLensHomeDir, "ratings.dat")).map(parseRating)

    # movies is an RDD of (movieId, movieTitle)
    movies = dict(sc.textFile(join(movieLensHomeDir, "movies.dat")).map(parseMovie).collect())

    # your code here
    numRatings = ratings.count()
    numUsers = ratings.values().map(lambda x: x[0]).distinct().count()
    numMovies = ratings.values().map(lambda x: x[1]).distinct().count()

    print "Got %d ratings from %d users on %d movies." % (numRatings, numUsers, numMovies)

    setSplits = ratings.randomSplit([6.0, 2.0, 2.0], 24)

    training = setSplits[0].values().union(myRatingsRDD).cache()
    validation = setSplits[1].values().cache()
    test = setSplits[2].values().cache()

    numTraining = training.count()
    numValidation = validation.count()
    numTest = test.count()


    print "Training: %d, validation: %d, test: %d" % (numTraining, numValidation, numTest)


    rank = 10
    model = ALS.train(training, rank, 5)

    # Evaluate the model
    testRmse = computeRmse(model, test, numTest)
    print "Our model has a RMSE of %s" % (testRmse)

    moviesToExcludeIds = set([x[1] for x in myRatings])
    candidates = sc.parallelize([m for m in movies if m not in moviesToExcludeIds]).map(lambda x: (0, x))
    print "This is a candidate %s" % candidates.take(1)

    predictions = sorted(model.predictAll(candidates).collect(), key=lambda x: x[2], reverse=True)[:50]


    for prediction in predictions:
        print movies[prediction[1]]


    # clean up
    sc.stop()

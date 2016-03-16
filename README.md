
# Movie Recommender Spark

This is an example of using machine learning spark library to predict the user movie ratings to recommend 50 movies to that user.

## Run

Execute python script to rate movies

`` ./rateMovies``

Execute Spark job to get the results (Execute inside the spark folder)

``
./bin/spark-submit python/MovieLensALS.py /PATH/TO/PROJECT/pythonMovies/data/movielens/medium /PATH/TO/PROJECT/pythonMovies/data/movielens/personalRatings.txt
``

### Tech
Recommender is based on the following technologies:

  - [ApacheSparkMl] - Apache Spark is a fast and general engine for large-scale data processing.

   [ApacheSpark]: <https://spark.apache.org/>

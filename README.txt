TO RUN:

1) Move the files zip_1000.train, zip_3000.train, zip_5000.train, zip_7291.train and zip.test to your hdfs input folder.
2) hadoop jar /local/dir/KnnPattern.jar KnnPattern /local/dir /hdfs/input/dir /hdfs/output/dir

Notes: 
- Dir paths should not have a trailing backslash.
- Results will be written to file knn_mapRed_results.txt
- Predictions tested against test_classes.txt (just a text file of the actual classes in the test data).

USAGE EXAMPLE:

hadoop fs -rm -r /user

hadoop fs -mkdir -p /user/artiles
hadoop fs -mkdir /user/artiles/input

hadoop fs -copyFromLocal /Users/oswaldoartiles/FinalProject.CAP5768/MapReduce-KNN/zip_1000.train /user/artiles/input
hadoop fs -copyFromLocal /Users/oswaldoartiles/FinalProject.CAP5768/MapReduce-KNN/zip_3000.train /user/artiles/input
hadoop fs -copyFromLocal /Users/oswaldoartiles/FinalProject.CAP5768/MapReduce-KNN/zip_5000.train /user/artiles/input
hadoop fs -copyFromLocal /Users/oswaldoartiles/FinalProject.CAP5768/MapReduce-KNN/zip_7291.train /user/artiles/input
hadoop fs -copyFromLocal /Users/oswaldoartiles/FinalProject.CAP5768/MapReduce-KNN/zip.test /user/artiles/input

hadoop jar /Users/oswaldoartiles/FinalProject.CAP5768/MapReduce-KNN/KnnPattern.jar KnnPattern /Users/oswaldoartiles/FinalProject.CAP5768/MapReduce-KNN /user/artiles/input /user/artiles/output
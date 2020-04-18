# KCF tracker with Kalman filter

## What?

Here lies my university course project

## Why?

I had an idea to increase accuracy and robustness of a famous KCF tracker. And I thought maybe a Kalman filter can do its filtering thing and place a bounding box more accurately. But I didn't want to select parameters of Kalman filter by hand, so I implemented some sort of genetic algorithm to do it for me.

## What Algorithm?

I used generic Kalman filter algorithm. The state vector was as follows: (x1, y1,), where (x1, y1) - is the center of the bounding box.

Genetic algorithm selected elements for B, u, S, R and A matrices. I used f-measure with beta = 3 as a fitness function. Other parameters of the fitness function were accuracy and robustness. They were calculated plain VOT way, although I used DIOU instead of IOU, and omit less then 10 frames before calculation of the metrics.

## Where is the best genome?

Here:



## Acknowledgments

[KCF repository](https://github.com/joaofaro/KCFcpp)

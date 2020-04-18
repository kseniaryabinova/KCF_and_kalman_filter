# KCF tracker with Kalman filter

## What?

Here lies my university course project

## Why?

I had an idea to increase accuracy and robustness of a famous KCF tracker. And I thought maybe a Kalman filter can do its filtering thing and place a bounding box more accurately. But I didn't want to select parameters of Kalman filter by hand, so I implemented some sort of genetic algorithm to do it for me.

## What Algorithm?

I used generic Kalman filter algorithm. The state vector was as follows: (x1, y1,), where (x1, y1) - is the center of the bounding box.

Genetic algorithm selected elements for B, u, S, R and A matrices. I used f-measure with beta = 3 as a fitness function. Other parameters of the fitness function were accuracy and robustness. They were calculated plain VOT way, although I used DIOU instead of IOU, and omit less then 10 frames before calculation of the metrics.

## Where is the best genome?

Here:git

-1.04996 -0.35197 2.14865 1.81949 1.0531 0.379274 -0.972355 -4.83614 -0.664763 1.58524 3.81422 0.125361 -0.805518 1.51277 -2.40536 -3.02137 2.54712 0.250593 0.310221 0.0344269 -0.52078 2.17617 0.0965927 0.910537 0.0220096 0.745542 -1.34083 0.2846 -1.66556 -1.72201 -0.333888 2.63724 1.7406 -2.1478 -0.25823 0.908437 3.43149 -0.00610241 -0.767901 0.0167775 1.48063 0.0126195 0.167397 -0.0533417 -0.0268063 -0.04304 -0.0294877 -2.30189 0.188723 -0.300192 1.17634 1.12888 0.0306505 0.0520604 -0.249356 -3.18886

## Acknowledgments

[KCF repository](https://github.com/joaofaro/KCFcpp)

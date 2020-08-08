#include <iostream>
#include <chrono>
#include <fstream>
#include <sstream>
#include <algorithm>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/tracking.hpp>

#include "kcftracker.hpp"

#include <dirent.h>

using namespace std;
using namespace cv;


void webcam_run(KCFTracker& tracker) {
	auto video = cv::VideoCapture(0);
	cv::namedWindow("webcam");

	cv::Mat frame;
	cv::Mat resized_frame;
	cv::Rect box;
	bool is_first = true;
	float factor = 1;
    std::chrono::high_resolution_clock::time_point start_time;
    std::chrono::high_resolution_clock::time_point finish_time;

	while (true){
		video.read(frame);

		if (frame.data == nullptr) {
			break;
		}

		cv::resize(frame, resized_frame, cv::Size(int(640/factor), int(480/factor)));
		auto key = cv::waitKey(1);
		if (key == 27){
			break;

		} else if (key == int('b') && is_first){
			box = cv::selectROI("webcam", resized_frame);

			start_time = std::chrono::high_resolution_clock::now();
			tracker.init(box, resized_frame);
            finish_time = std::chrono::high_resolution_clock::now();

			printf("init time=%ld\n",
			        std::chrono::duration_cast<std::chrono::milliseconds>(finish_time - start_time).count());

			is_first = false;

		} else if (!is_first){
            start_time = std::chrono::high_resolution_clock::now();
			box = tracker.update(resized_frame);
            finish_time = std::chrono::high_resolution_clock::now();

            printf("update time=%ld\n",
                    std::chrono::duration_cast<std::chrono::milliseconds>(finish_time - start_time).count());

//			box = cv::Rect(int(box.x * factor), int(box.y * factor),
//                           int(box.width * factor), int(box.height * 2));

//			printf("after update {%d, %d, %d, %d}\n", box.x, box.y, box.width, box.height);
			cv::rectangle(resized_frame, box, cv::Scalar(0, 0, 255), 2);
		}

		cv::imshow("webcam", resized_frame);
	}

	video.release();

}


int main(int argc, char* argv[]){

	if (argc > 5) return -1;

	bool HOG = true;
	bool FIXEDWINDOW = false;
	bool MULTISCALE = true;
	bool SILENT = true;
	bool LAB = false;

	for(int i = 0; i < argc; i++){
		if ( strcmp (argv[i], "hog") == 0 )
			HOG = true;
		if ( strcmp (argv[i], "fixed_window") == 0 )
			FIXEDWINDOW = true;
		if ( strcmp (argv[i], "singlescale") == 0 )
			MULTISCALE = false;
		if ( strcmp (argv[i], "show") == 0 )
			SILENT = false;
		if ( strcmp (argv[i], "lab") == 0 ){
			LAB = true;
			HOG = true;
		}
		if ( strcmp (argv[i], "gray") == 0 )
			HOG = false;
	}
	
	// Create KCFTracker object
	KCFTracker tracker(HOG, FIXEDWINDOW, MULTISCALE, LAB);

	// Frame readed
	Mat frame;

	// Tracker results
	Rect result;

	webcam_run(tracker);
}

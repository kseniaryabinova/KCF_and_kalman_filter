#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "../include/kcftracker.hpp"

#include <dirent.h>
#include <kalman_filter.h>

#include <chrono>

using namespace std;
using namespace cv;
using namespace chrono;


void webcam_run(KCFTracker& tracker) {
	auto video = cv::VideoCapture(0);
	cv::namedWindow("webcam");

	cv::Mat frame;
	cv::Rect box;
	bool is_first = true;

	Kalman kalman;
    float genome[] = {
            -1.04996, -0.35197, 2.14865, 1.81949, 1.0531,
            0.379274, -0.972355, -4.83614, -0.664763, 1.58524,
            3.81422, 0.125361, -0.805518, 1.51277, -2.40536,
            -3.02137, 2.54712, 0.250593, 0.310221, 0.0344269,
            -0.52078, 2.17617, 0.0965927, 0.910537, 0.0220096,
            0.745542, -1.34083, 0.2846, -1.66556, -1.72201,
            -0.333888, 2.63724, 1.7406, -2.1478, -0.25823,
            0.908437, 3.43149, -0.00610241, -0.767901, 0.0167775,
            1.48063, 0.0126195, 0.167397, -0.0533417, -0.0268063,
            -0.04304, -0.0294877, -2.30189, 0.188723, -0.300192,
            1.17634, 1.12888, 0.0306505, 0.0520604, -0.249356, -3.18886
    };
    kalman.set_from_genome(genome);
	microseconds T;

	while (true){
		video.read(frame);

		if (frame.data == nullptr) {
			break;
		}

		auto key = cv::waitKey(1);
		if (key == 27){
			break;

		} else if (key == int('b') && is_first){
			box = cv::selectROI("webcam", frame);
			tracker.init(box, frame);
			printf("after init\n");

			is_first = false;

		} else if (!is_first){
			T = duration_cast<microseconds>(system_clock::now().time_since_epoch());

			box = tracker.update(frame);
			cv::rectangle(frame, box, cv::Scalar(0, 0, 255), 3);

			auto T_new = duration_cast<microseconds>(system_clock::now().time_since_epoch());
			auto kalman_box = kalman.predict(float((T_new - T).count()) / 1'000'000, box);

			cv::rectangle(frame, kalman_box, cv::Scalar(0, 255, 0), 3);

			printf("after update {%d, %d, %d, %d}  ---  {%d, %d, %d, %d}\n",
					box.x, box.y, box.width, box.height,
				    kalman_box.x, kalman_box.y, kalman_box.width, kalman_box.height
					);
		}

		cv::imshow("webcam", frame);
	}

	video.release();
    cv::destroyAllWindows();
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

	// Path to list.txt
//	ifstream listFile;
//	string fileName = "images.txt";
//  	listFile.open(fileName);
//
//  	// Read groundtruth for the 1st frame
//  	ifstream groundtruthFile;
//	string groundtruth = "region.txt";
//  	groundtruthFile.open(groundtruth);
//  	string firstLine;
//  	getline(groundtruthFile, firstLine);
//	groundtruthFile.close();
//
//  	istringstream ss(firstLine);
//
//  	// Read groundtruth like a dumb
//  	float x1, y1, x2, y2, x3, y3, x4, y4;
//  	char ch;
//	ss >> x1;
//	ss >> ch;
//	ss >> y1;
//	ss >> ch;
//	ss >> x2;
//	ss >> ch;
//	ss >> y2;
//	ss >> ch;
//	ss >> x3;
//	ss >> ch;
//	ss >> y3;
//	ss >> ch;
//	ss >> x4;
//	ss >> ch;
//	ss >> y4;
//
//	// Using min and max of X and Y for groundtruth rectangle
//	float xMin =  min(x1, min(x2, min(x3, x4)));
//	float yMin =  min(y1, min(y2, min(y3, y4)));
//	float width = max(x1, max(x2, max(x3, x4))) - xMin;
//	float height = max(y1, max(y2, max(y3, y4))) - yMin;
//
//
//	// Read Images
//	ifstream listFramesFile;
//	string listFrames = "images.txt";
//	listFramesFile.open(listFrames);
//	string frameName;
//
//
//	// Write Results
//	ofstream resultsFile;
//	string resultsPath = "output.txt";
//	resultsFile.open(resultsPath);
//
//	// Frame counter
//	int nFrames = 0;
//
//	while ( getline(listFramesFile, frameName) ){
//		frameName = frameName;
//
//		// Read each frame from the list
//		frame = imread(frameName, CV_LOAD_IMAGE_COLOR);
//
//		// First frame, give the groundtruth to the tracker
//		if (nFrames == 0) {
//			tracker.init( Rect(xMin, yMin, width, height), frame );
//			rectangle( frame, Point( xMin, yMin ), Point( xMin+width, yMin+height), Scalar( 0, 255, 255 ), 1, 8 );
//			resultsFile << xMin << "," << yMin << "," << width << "," << height << endl;
//		}
//		// Update
//		else{
//			result = tracker.update(frame);
//			rectangle( frame, Point( result.x, result.y ), Point( result.x+result.width, result.y+result.height), Scalar( 0, 255, 255 ), 1, 8 );
//			resultsFile << result.x << "," << result.y << "," << result.width << "," << result.height << endl;
//		}
//
//		nFrames++;
//
//		if (!SILENT){
//			imshow("Image", frame);
//			waitKey(1);
//		}
//	}
//	resultsFile.close();
//
//	listFile.close();

}

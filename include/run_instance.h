#ifndef TEST_RUN_INSTANCE_H
#define TEST_RUN_INSTANCE_H


#include <string>
#include <vector>
#include <experimental/filesystem>
#include <algorithm>
#include <fstream>
#include <iterator>

using namespace std::experimental::filesystem;

class Statistics{
public:
    Statistics (const std::string &path_to_dirs, const int pref){
        bboxes_info = new std::vector<std::string>();
        time_info = new std::vector<std::string>();
        file_names = new std::vector<path>();
        dirs = new std::vector<path>();
        is_new_video = true;
        prefix = std::to_string(pref);

        for(auto& p : directory_iterator(path_to_dirs)) {
            if (p.status().type() == file_type::directory) {
                dirs->push_back(p.path());
            }
        }
        make_file_names(dirs->back());
    }

    bool check_is_new_video(){
        return is_new_video;
    }

    bool try_get_next_file(std::string& path_to_image){
        is_new_video = false;

        if (file_names->empty()){
            make_bboxes_file(dirs->back());
            printf("--- %s\n", dirs->back().c_str());
            dirs->pop_back();
            if (dirs->empty()){
                return false;
            }
            make_file_names(dirs->back());
        }
        if (bboxes_info->empty())
            is_new_video = true;

        path_to_image = file_names->back().string();
        file_names->pop_back();

        return true;
    }

    void bboxes_to_file(int x1, int y1, int x2, int y2){
        bboxes_info->push_back(std::to_string(x1) + ";" +
                               std::to_string(y1) + ";" +
                               std::to_string(x2) + ";" +
                               std::to_string(y2));
    }

    std::vector<int>* read_current_groundtruth(){
        std::vector<int>* coords = new std::vector<int>(4);
        std::fstream groundtruth(dirs->back() / "groundtruth.txt");
        std::string str;
        std::getline(groundtruth, str);

        std::stringstream ss(str);
        std::vector<int> f_c;

        while( ss.good() ) {
            std::string substr;
            getline( ss, substr, ',' );
            f_c.push_back(std::stoi(substr));
        }

        if (f_c.size() > 4){
            coords->at(0) = int(fmin(f_c[0], fmin(f_c[2], fmin(f_c[4], f_c[6]))));
            coords->at(1) = int(fmin(f_c[1], fmin(f_c[3], fmin(f_c[5], f_c[7]))));
            coords->at(2) = int(fmax(f_c[0], fmax(f_c[2], fmax(f_c[4], f_c[6])))) - coords->at(0);
            coords->at(3) = int(fmax(f_c[1], fmax(f_c[3], fmax(f_c[5], f_c[7])))) - coords->at(1);
        } else {
            coords->at(0) = int(f_c[0]);
            coords->at(1) = int(f_c[1]);
            coords->at(2) = int(f_c[2]);
            coords->at(3) = int(f_c[3]);
        }

        return coords;
    }

protected:
    void make_bboxes_file(const path& dir){
        if (bboxes_info->empty())
            return;

        std::ofstream file(path_to_bboxes_dir /
                           (prefix + dir.filename().string() + ".csv"));

        std::ostream_iterator<std::string> out_it (file,"\n");
        std::copy ( bboxes_info->begin(), bboxes_info->end(), out_it );
        file.flush();
        file.close();
        bboxes_info->clear();
    }

    void make_file_names(const path &dir){
        for (auto& file : directory_iterator(dir)){
            if (file.status().type() == file_type::regular &&
                file.path().extension() == ".jpg"){

                file_names->push_back(file.path());
            }
        }
        sort(file_names->begin(), file_names->end(),
             [](path a, path b) { return a > b; });
    }

private:
    bool is_new_video;
    std::string path_to_bboxes_dir = "/home/ksenia/CLionProjects/bboxes_info";
    std::string prefix;
    std::vector<std::string>* bboxes_info;
    std::vector<std::string>* time_info;
    std::vector<path>* file_names;
    std::vector<path>* dirs;
};

#include "genetic_algorithm.h"
#include "kcftracker.hpp"

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>

void run_statistics(genetic_alg::Population& population) {

    cv::Mat frame;
    cv::Rect result;

    for (auto &person : population.people) {

        KCFTracker tracker(true, false, true, false);

        Statistics stat("/media/ksenia/dataset", person->get_number());
        std::string file_path;

        while (stat.try_get_next_file(file_path)) {
            frame = cv::imread(file_path, CV_LOAD_IMAGE_COLOR);

            if (stat.check_is_new_video()) {
                auto coords = stat.read_current_groundtruth();
                tracker.init(cv::Rect(coords->at(0), coords->at(1),
                                      coords->at(2), coords->at(3)), frame);
            }

            result = tracker.update(frame);
            stat.bboxes_to_file(result.x, result.y, result.width, result.height);

            rectangle(frame, cv::Point(result.x, result.y),
                      cv::Point(result.x + result.width, result.y + result.height),
                      cv::Scalar(0, 255, 255), 4, 8);
            imshow("Image", frame);
            if (27 == cv::waitKey(1)) {
                break;
            }
        }
    }
}


#endif //TEST_RUN_INSTANCE_H

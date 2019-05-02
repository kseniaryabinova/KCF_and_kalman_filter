#include <memory>

#include <memory>

#ifndef TEST_RUN_INSTANCE_H
#define TEST_RUN_INSTANCE_H


#include <string>
#include <vector>
#include <experimental/filesystem>
#include <algorithm>
#include <fstream>
#include <iterator>
#include <unordered_map>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <bits/unordered_map.h>
//#include <omp.h>


using namespace std::experimental::filesystem;

class Statistics{
public:
    Statistics (const std::string &path_to_dirs) : path_to_dirs(path_to_dirs){
        dirs = new std::vector<path>();

        for(auto& p : directory_iterator(path_to_dirs)) {
            if (p.status().type() == file_type::directory) {
                dirs->push_back(p.path());
            }
        }

        read_all_groundtruth();
    }

    void init (const int pref){
        bboxes_info = new std::vector<std::string>();
        file_names = new std::vector<path>();
        dirs = new std::vector<path>();
        is_new_video = true;
        prefix = std::to_string(pref);
        gt_index = 0;

        if (dirs->empty()){
            for(auto& p : directory_iterator(path_to_dirs)) {
                if (p.status().type() == file_type::directory) {
                    dirs->push_back(p.path());
                }
            }
        }

        make_file_names(dirs->back());
    }


    double iou(const cv::Rect& box1, const cv::Rect& box2){
        double x1 = std::max(box1.x, box2.x);
        double y1 = std::max(box1.y, box2.y);
        double x2 = std::min(box1.x + box1.width,  box2.x + box2.width);
        double y2 = std::min(box1.y + box1.height, box2.y + box2.height);

        if (x1 >= x2 || y1 >= y2){
            return 0;
        }

        double intersection = (x2 - x1) * (y2 - y1);
        double union_ = box1.width * box1.height + box2.width * box2.height - intersection;

        return intersection / union_;
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
            gt_index = 0;
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

    void bboxes_to_file(const cv::Rect& tr_box, const double iou_value){
        bboxes_info->push_back(std::to_string(tr_box.x) + ";" +
                               std::to_string(tr_box.y) + ";" +
                               std::to_string(tr_box.width) + ";" +
                               std::to_string(tr_box.height) + ";" +
                               std::to_string(iou_value));
    }

    void read_all_groundtruth(){
        std::vector<int> f_c;

        printf("start to read gt files\n");

        for (int i=0; i<dirs->size(); ++i){
            printf("%d of %lu\n", i+1, dirs->size());

            auto rects = std::vector<cv::Rect>();
            std::string str;

            std::fstream gt_file(dirs->at(i) / "groundtruth.txt");

            while (std::getline(gt_file, str)){
                f_c.clear();
                std::stringstream substr_stream(str);

                while (substr_stream.good()){
                    std::string substr;
                    std::getline(substr_stream, substr, ',');
                    f_c.push_back(std::stoi(substr));
                }

                cv::Rect box;

                if (f_c.size() > 4){
                    box.x = int(fmin(f_c[0], fmin(f_c[2], fmin(f_c[4], f_c[6]))));
                    box.y = int(fmin(f_c[1], fmin(f_c[3], fmin(f_c[5], f_c[7]))));
                    box.width = int(fmax(f_c[0], fmax(f_c[2], fmax(f_c[4], f_c[6])))) - box.x;
                    box.height = int(fmax(f_c[1], fmax(f_c[3], fmax(f_c[5], f_c[7])))) - box.y;
                } else {
                    box.x = int(f_c[0]);
                    box.y = int(f_c[1]);
                    box.width = int(f_c[2]);
                    box.height = int(f_c[3]);
                }

                rects.push_back(box);
            }

            groundtruth[dirs->at(i).filename().string()] = rects;
        }
    }

    cv::Rect read_init_groundtruth(){
        if (!gt_index){
            auto box = groundtruth[dirs->back().filename().string()][gt_index+1];
            printf("%d %d %d %d\n", box.x, box.y, box.width, box.height);
            return groundtruth[dirs->back().filename().string()][gt_index++];
        } else {
            auto box = groundtruth[dirs->back().filename().string()][0];
            printf("%d %d %d %d\n", box.x, box.y, box.width, box.height);
            return groundtruth[dirs->back().filename().string()][gt_index++];
        }
    }

    cv::Rect read_current_groundtruth(){
        auto box = groundtruth[dirs->back().filename().string()][gt_index+1];
        printf("\t%d %d %d %d\n", box.x, box.y, box.width, box.height);
        return groundtruth[dirs->back().filename().string()][gt_index++];
    }

protected:
    void make_bboxes_file(const path& dir){
        if (bboxes_info->empty())
            return;

        std::ofstream file(path_to_bboxes_dir /
                           (prefix + "_" + dir.filename().string() + ".csv"));

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
    std::unordered_map<std::string, std::vector<cv::Rect>> groundtruth;
    int gt_index;

    bool is_new_video;
    std::string path_to_bboxes_dir = "/home/ksenia/bboxes_info";
    std::string prefix;
    std::vector<std::string>* bboxes_info;
    std::vector<path>* file_names;
    std::vector<path>* dirs;

    std::string path_to_dirs;
};

#include "genetic_algorithm.h"
#include "kcftracker.hpp"
#include "kalman_filter.h"

void run_statistics(genetic_alg::Population& population) {

    cv::Mat frame;
    cv::Rect result;
    Statistics stat("/media/ksenia/4C62-AB81/vot2017");
    double iou = 0;

    printf("start to run population\n");

    for (auto &person : population.people) {

        printf("%d of %d\n", person->get_number(), person->counter);

        stat.init(person->get_number());
        std::unique_ptr<KCFTracker> tracker;
        std::unique_ptr<Kalman> kalman;

        std::string file_path;

        while (stat.try_get_next_file(file_path)) {
            frame = cv::imread(file_path, CV_LOAD_IMAGE_COLOR);

            if (stat.check_is_new_video() || iou == 0) {
                cv::Rect coords = stat.read_init_groundtruth();
                tracker = std::make_unique<KCFTracker>(true, false, true, false);
                tracker->init(coords, frame);

                iou = 1;

//                rectangle(frame, cv::Point(coords.x, coords.y),
//                          cv::Point(coords.x + coords.width, coords.y + coords.height),
//                          cv::Scalar(0, 255, 0), 4, 8);
//                kalman = std::make_unique<Kalman>();
            } else {
                result = tracker->update(frame);

                iou = stat.iou(result, stat.read_current_groundtruth());

//                rectangle(frame, cv::Point(result.x, result.y),
//                          cv::Point(result.x + result.width, result.y + result.height),
//                          cv::Scalar(0, 255, 255), 4, 8);
            }

            stat.bboxes_to_file(result, iou);

//            imshow("Image", frame);
//            if (27 == cv::waitKey(0)) {
//                return;
//            }
        }
    }
}


#endif //TEST_RUN_INSTANCE_H

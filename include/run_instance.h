#include <utility>

#ifndef TEST_RUN_INSTANCE_H
#define TEST_RUN_INSTANCE_H

#include <string>
#include <memory>
#include <vector>
#include <experimental/filesystem>
#include <algorithm>
#include <fstream>
#include <iterator>
#include <unordered_map>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <bits/unordered_map.h>


using namespace std::experimental::filesystem;

std::unordered_map<std::string, std::vector<cv::Rect>> groundtruth;

void read_all_groundtruth(const std::string &path_to_dirs){
    std::vector<int> f_c;
    std::string str;
    std::string substr;

    std::vector<path> dirs;
    for(auto& p : directory_iterator(path_to_dirs)) {
        if (p.status().type() == file_type::directory) {
            dirs.push_back(p.path());
        }
    }

    printf("start to read gt files\n");

    for (int i=0; i<dirs.size(); ++i){
        printf("%d of %lu\n", i+1, dirs.size());

        auto rects = std::vector<cv::Rect>();

        std::fstream gt_file(dirs[i] / "groundtruth.txt");

        while (std::getline(gt_file, str)){
            f_c.clear();
            std::stringstream substr_stream(str);

            while (std::getline(substr_stream, substr, ',')){
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

        groundtruth[dirs[i].filename().string()] = rects;
    }
}

class Statistics{
public:
    Statistics (std::string path_to_dirs, const int pref) : path_to_dirs(std::move(path_to_dirs)){
        dirs = new std::vector<path>();
        bboxes_info = new std::vector<std::string>();
        file_names = new std::vector<path>();
        dirs = new std::vector<path>();
        is_new_video = true;
        prefix = std::to_string(pref);
        gt_index = 0;

        if (dirs->empty()){
            for(auto& p : directory_iterator(this->path_to_dirs)) {
                if (p.status().type() == file_type::directory) {
                    dirs->push_back(p.path());
                }
            }
        }

        create_directory(path_to_bboxes_dir / prefix);
        make_file_names(dirs->back());
    }


    double giou(const cv::Rect& box1, const cv::Rect& box2){
        double x1 = std::max(box1.x, box2.x);
        double y1 = std::max(box1.y, box2.y);
        double x2 = std::min(box1.x + box1.width,  box2.x + box2.width);
        double y2 = std::min(box1.y + box1.height, box2.y + box2.height);

        double intersection = std::max(x2 - x1, 0.) * std::max(y2 - y1, 0.);
        double union_ = box1.width * box1.height + box2.width * box2.height - intersection;

        double area_1 = box1.height * box1.width;
        double area_2 = box2.height * box2.width;

        double x1_c = std::min(box1.x, box2.x);
        double y1_c = std::min(box1.y, box2.y);
        double x2_c = std::max(box1.x + box1.width,  box2.x + box2.width);
        double y2_c = std::max(box1.y + box1.height, box2.y + box2.height);
        double area_c = (x2_c - x1_c) * (y2_c - y1_c);

        double giou = intersection / union_ - (area_c - union_) / area_c;

        return giou;
    }

    bool check_is_new_video(){
        return is_new_video;
    }

    bool try_get_next_file(std::string& path_to_image){
        is_new_video = false;

        if (file_names->empty()){
            make_bboxes_file(dirs->back());
//            printf("--- %s\n", dirs->back().c_str());
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
        bboxes_info->push_back(std::to_string(iou_value));
    }

    cv::Rect read_current_groundtruth(){
        return groundtruth[dirs->back().filename().string()][gt_index++];
    }

protected:
    void make_bboxes_file(const path& dir){
        if (bboxes_info->empty())
            return;

        std::ofstream file(path_to_bboxes_dir / prefix /
                           (dir.filename().string() + ".csv"));

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
    int gt_index;

    bool is_new_video;
    std::string path_to_bboxes_dir = "../bboxes_info";
    std::string prefix;
    std::vector<std::string>* bboxes_info;
    std::vector<path>* file_names;
    std::vector<path>* dirs;

    std::string path_to_dirs;
};

#include "genetic_algorithm.h"
#include "kcftracker.hpp"
#include "kalman_filter.h"
#include <chrono>
#include <omp.h>

using namespace std::chrono;
typedef steady_clock timestamp;


void run_statistics(genetic_alg::Population& population,
        const std::string& path_to_vids,
        int threads_amount = 8) {

    bool show = false;
    int frames_to_kalman = 50000;
    printf("start to run population\n");

    omp_set_dynamic(0);
    omp_set_num_threads(threads_amount);

#pragma omp parallel for
    for (int i= 0; i<population.people.size(); ++i){

        if (population.people[i]->p != -1 and !population.people[i]->is_mutated()){
            continue;
        }

        cv::Mat frame;
        cv::Rect result;
        double iou = 0;
        int kalman_counter = 0;
        auto T = timestamp::now();

        printf("%d of %zu\n",
                population.people[i]->get_number(),
                population.people.size());

        Statistics stat(path_to_vids, population.people[i]->get_number());
        std::unique_ptr<KCFTracker> tracker;
        std::unique_ptr<Kalman> kalman;

        std::string file_path;

        while (stat.try_get_next_file(file_path)) {
            frame = cv::imread(file_path, CV_LOAD_IMAGE_COLOR);

            if (stat.check_is_new_video() || iou <= 0 || kalman_counter > frames_to_kalman) {
                auto coords = stat.read_current_groundtruth();
                tracker = std::make_unique<KCFTracker>(
                        true,false, true, false);
                tracker->init(coords, frame);

                kalman = std::make_unique<Kalman>();
//                printf("\nperson #%d: ", population.people[i]->get_number());
//                for (int j=0; j<genetic_alg::GENOME_LENGTH; ++j){
//                    printf("[%d]=%f ", j, population.people[i]->data[j]);
//                }
                kalman->set_from_genome(population.people[i]->data);

                if (show)
                    rectangle(frame, cv::Point(coords.x, coords.y),
                            cv::Point(coords.x + coords.width, coords.y + coords.height),
                            cv::Scalar(0, 255, 0), 4, 8);

                iou = 1;
                kalman_counter = 0;

            } else {
                T = timestamp::now();
                result = tracker->update(frame);

                if (show)
                    rectangle(frame, cv::Point(result.x, result.y),
                              cv::Point(result.x + result.width, result.y + result.height),
                              cv::Scalar(0, 255, 255), 4, 8);

                result = kalman->predict(
                        double(duration_cast<microseconds>(timestamp::now() - T).count()) / 1000'000.,
                        result);

                if (show)
                    rectangle(frame, cv::Point(result.x, result.y),
                              cv::Point(result.x + result.width, result.y + result.height),
                              cv::Scalar(255, 0, 255), 4, 8);

                iou = stat.giou(result, stat.read_current_groundtruth());
                ++kalman_counter;
            }

            stat.bboxes_to_file(result, iou);

            if (show){
                imshow("Image", frame);
//                if (27 == cv::waitKey(3)) {
//                    return;
//                }
            }
        }
    }

    // -----------------------------------------------------------------------------------------

    printf("start selection\n");

    for (auto& person : population.people){
        person->count_fitness();
    }

    double mean_sum = 0;
    for (auto& person : population.people){
        mean_sum += person->fitness_value;
    }
    double mean = mean_sum / population.people.size();

    double variance_delta_sum = 0;
    for (auto& person : population.people){
        variance_delta_sum += (person->fitness_value - mean) * (person->fitness_value - mean);
    }
    double variance = variance_delta_sum / (population.people.size() - 1);
    double standart_derivation = sqrt(variance);

    double F_i_sum = 0;
    for (auto& person : population.people){
        F_i_sum += person->count_F_i(standart_derivation, mean);
    }

    for (auto& person : population.people){
        person->count_probability(F_i_sum);
        person->log_info();
    }

    // -----------------------------------------------------------------------------------------

    printf("start create new population\n");
    population.create_new_popuation();
}


#endif //TEST_RUN_INSTANCE_H

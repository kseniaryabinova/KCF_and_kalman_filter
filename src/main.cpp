#include <string>

#include "genetic_algorithm.h"
#include "run_instance.h"
#include "../include/genetic_algorithm.h"


int main(){
    std::string path_to_videos = "../vot2017";
    read_all_groundtruth(path_to_videos);

    std::string genome = "1.5999 0.550499 0.161915 0.182506 0.20654 -0.13663 -0.459912 0.0801636 0.892961 -0.101684 -1.02445 0.692917 0.403214 1.44336 0.908085 -0.00933981 -0.0774465 1.80309 0.950267 0.665119 -0.667088 0.0206512 0.580625 -2.09835 0.500194 -0.737744 1.41335 0.8887 0.0515018 0.64397 0.201066 -0.503415 0.460445 1.57769 1.25405 0.217351 0.832492 0.990456 -1.88612 0.451881 0.067441 -0.00365546 2.04729 1.64105 -0.00091322 -0.168288 1.20953 0.585769 0.459155 1.6227 2.81636 1.02016 0.702955 -0.313392 2.25244 -0.148233";

    auto person = genetic_alg::Genome(genome);

    run_statistics(person, path_to_videos);
    return 0;
}

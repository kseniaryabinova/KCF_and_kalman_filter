#include <string>

#include "genetic_algorithm.h"
#include "run_instance.h"


int main(){
    std::string path_to_videos = "../vot2017";
    read_all_groundtruth(path_to_videos);
    genetic_alg::Population population;

    for (int i=0; i<10; ++i){
        run_statistics(population, path_to_videos);
    }
    return 0;
}

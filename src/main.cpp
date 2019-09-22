#include "genetic_algorithm.h"
#include "run_instance.h"


int main(){
    read_all_groundtruth("../vot2017");
    genetic_alg::Population population;

    for (int i=0; i<20; ++i){
        run_statistics(population);
    }
    return 0;
}

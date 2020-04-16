#ifndef TEST_GENETIC_ALGORITHM_H
#define TEST_GENETIC_ALGORITHM_H


#include <ctime>
#include <cmath>
#include <random>
#include <memory>
#include <vector>
#include <experimental/filesystem>
#include <fstream>
#include <algorithm>

namespace genetic_alg{

    using namespace std::experimental::filesystem;

    const int GENOME_LENGTH = 16 + 4 + 16 + 4 + 16;

    std::fstream fitness_log_file("../fitness.log", std::fstream::out | std::fstream::trunc);
    std::fstream genome_log_file("../genome.log", std::fstream::out | std::fstream::trunc);

    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_real_distribution<float> init_rand(0., 1.);
    std::uniform_real_distribution<float> mutate_rand(-1., 1.);
    std::uniform_int_distribution<int> crossingover_dist_lo(2, GENOME_LENGTH / 2 - 1);
    std::uniform_int_distribution<int> crossingover_dist_hi(GENOME_LENGTH / 2 + 1, GENOME_LENGTH - 1);

    static const int MAX_FAIL_COUNTER = 40'000;

    class Genome{
    public:

        typedef std::pair<std::shared_ptr<Genome>, std::shared_ptr<Genome>> children;

        static int counter;
        float* data;

        ~Genome(){
            delete[] this->data;
        }

        Genome(bool is_random = false) {
            this->data = new float[GENOME_LENGTH];

            number = ++counter;

            if (is_random){
                for (int i=0; i<GENOME_LENGTH; ++i) {
                    this->data[i] = init_rand(mt);
                }
            } else {
                for (int i=0; i<GENOME_LENGTH; ++i) {
                    this->data[i] = 0;
                }
            }
        }

        Genome(std::string& genome_string){
            this->data = new float[GENOME_LENGTH];
            number = ++counter;

            // genes are separated by space
            std::string delimiter = " ";

            auto start = 0U;
            auto end = genome_string.find(delimiter);
            int i = 0;

            while (end != std::string::npos){
                float gene = std::stof(genome_string.substr(start, end - start));
                this->data[i++] = gene;

                start = end + delimiter.length();
                end = genome_string.find(delimiter, start);
            }

            this->data[i] = std::stof(genome_string.substr(start, genome_string.size() - start));
        }

        int get_number(){
            return number;
        }

        bool is_mutated(){
            return this->mutated;
        }

        void set_mutated(bool mutated_){
            this->mutated = mutated_;
        }

        children make_kids_with(const std::shared_ptr <Genome> &that) {
            int positions[4] = {0, crossingover_dist_lo(mt),
                                crossingover_dist_hi(mt), GENOME_LENGTH};

            auto child_1 = std::make_unique<Genome>();
            auto child_2 = std::make_unique<Genome>();

            // crossingover in 2 places
            for (int i = 0; i < 3; ++i) {
                if (init_rand(mt) > 0.5) {
                    for (int j = positions[i]; j < positions[i + 1]; ++j) {
                        child_1->data[j] = this->data[j];
                        child_2->data[j] = that->data[j];
                    }
                } else {
                    for (int j = positions[i]; j < positions[i + 1]; ++j) {
                        child_1->data[j] = that->data[j];
                        child_2->data[j] = this->data[j];
                    }
                }
            }

            return children(std::move(child_1), std::move(child_2));
        }

        double get_distance(const std::shared_ptr<Genome>& that){
            static const float DIFFERENCE_THRESHOLD = 0.01;
            double distance = 0;

            for (int i=0; i<GENOME_LENGTH; ++i){
                if (std::abs(this->data[i] - that->data[i]) > DIFFERENCE_THRESHOLD) {
                    ++distance;
                }
            }

            return distance;
        }

        void mutate(){
            this->mutated = true;
            double threshold = init_rand(mt);

            for (int i=0; i<GENOME_LENGTH; ++i) {
                if (init_rand(mt) > threshold){
                    this->data[i] += mutate_rand(mt);
                }
            }
        }

        void count_fitness(){
            double iou_sum = 0;
            long iou_counter = 0;
            double current_iou = 0;
            int first_n_counter = 0;

            long fail_counter = 0;

            std::string str;

            for (auto& file_path: directory_iterator(path_to_bboxes_dir / std::to_string(number))){
                std::fstream boxes_file(file_path.path().string());

                while (std::getline(boxes_file, str)){
                    current_iou = std::stod(str);

                    if (current_iou <= 0){
                        ++fail_counter;
                        iou_sum += current_iou;
//                    } else if (current_iou == 1.){
//                        first_n_counter = 1;
//                    } else if (first_n_counter < 1){
//                        ++first_n_counter;
                    } else {
//                        ++iou_counter;
                        iou_sum += current_iou;
                    }
                }
            }

            this->robustness = double(MAX_FAIL_COUNTER - fail_counter) / double(MAX_FAIL_COUNTER) * 100;
            if (iou_counter == 0){
                this->accuracy = 0;
            } else {
                this->accuracy = iou_sum / double(iou_counter) * 100;
            }
//            fitness_value = 1/(1/robustness + 1/(accuracy * MAX_FAIL_COUNTER / 100));
            double beta = 3;
            fitness_value = (1 + beta*beta)*(
                    (this->robustness * this->accuracy)/(beta*beta*this->accuracy + this->robustness));
        }

        double count_F_i(double standart_derivation, double mean){
            F_i = 1 + (fitness_value - mean)/(2 * standart_derivation);
            return F_i;
        }

        void count_probability(double F_i_sum){
            p = F_i / F_i_sum;
        }

        void log_info(){
            fitness_log_file << "person #" << this->get_number() <<
                        " acc=" << this->accuracy <<
                        " rob=" << this->robustness <<
                        " fit=" << this->fitness_value <<
                        " p=" << this->p << std::endl;
            fitness_log_file.flush();

            genome_log_file << "person #" << this->get_number() << std::endl;
            for (int i=0; i<GENOME_LENGTH; ++i){
                genome_log_file << this->data[i] << " ";
            }
            genome_log_file << std::endl;
            genome_log_file.flush();

            printf("\t---- info has been logged ----\n");
        }

        double fitness_value = -1;
        double p = -1;

        double robustness = -1;
        double accuracy = -1;

    private:
        int number;
        double F_i = -1;
        bool mutated = false;

        std::string path_to_bboxes_dir = "../bboxes_info";
    };

    int Genome::counter = 0;



    const int MIN_AMOUNT = 100;
    const int MAX_AMOUNT = 120;

    using People = std::vector<std::shared_ptr<Genome>>;


    class Population{
    public:
        Population() {
            fitness_log_file.precision(8);
            std::srand ((unsigned int)(time(nullptr) / 2));
            int i = 0;

            if (exists(this->genomes_to_load_path)){
                auto genomes_file = std::fstream(this->genomes_to_load_path);

                std::string line;
                while (std::getline(genomes_file, line)){
                    ++i;
                    people.emplace_back(std::make_unique<Genome>(line));
                }
            }

            for (; i < (MIN_AMOUNT + MAX_AMOUNT) / 2; ++i){
                people.emplace_back(std::make_unique<Genome>(true));
            }
        }

        std::pair<int, std::shared_ptr<Genome>> find_partner(const std::shared_ptr<Genome> &person) {
            int index = 0;
            double max_distance = 0;

            // FIXME check if people are the same

            for (int i = 0; i < people.size(); ++i) {
                if (person->get_number() != people[i]->get_number()){
                    double distance = people[i].get()->get_distance(person);

                    if (max_distance < distance) {
                        max_distance = distance;
                        index = i;
                    }
                }
            }

            return std::make_pair(max_distance, people[index]);
        }


        void create_new_popuation(){
            this->ancestors.insert(this->ancestors.end(), this->people.begin(), this->people.end());

            printf("reset mutated property to false\n");
            for (auto&& person : this->people){
                person->set_mutated(false);
            }

            People new_population;

            printf("sort people by fitness in descending order\n");
            std::sort(this->people.begin(), this->people.end(),
                    [](const std::shared_ptr<Genome>& a, const std::shared_ptr<Genome>& b){
                return a->fitness_value > b->fitness_value;
            });

            printf("copy good people\n");
            double fitness_sum = 0;
            for (auto&& person : this->people){
                fitness_sum += person->fitness_value;
            }

            for (int i=0; new_population.size() < MIN_AMOUNT / 4; ++i) {
                double limit = std::round(this->people[i]->fitness_value / fitness_sum * this->people.size() / 2);
                if (limit == 0) {
                    break;
                }

                for (int j=0; j<limit; ++j){
                    new_population.push_back(this->people[i]);
                }
            }

            printf("remove top 2 people from crossingover\n");
            this->people.erase(this->people.begin(), this->people.begin() + 2);

            printf("get potential partners\n");
            double mean = 1. / people.size();
            People thresholded_people;
            People people_after_selection;
            for (auto& person : people){
                if (person->p >= mean){
                    people_after_selection.push_back(person);
                } else {
                    thresholded_people.push_back(person);
                }
            }

            printf("make some children\n");
            const int DIVERSITY_THRESHOLD = 10;
            for (auto& person : people_after_selection){
                auto distance_and_partner = this->find_partner(person);

                if (distance_and_partner.first <= DIVERSITY_THRESHOLD){
                    std::shared_ptr<Genome> smaller_fitness_partner;
                    std::shared_ptr<Genome> bigger_fitness_partner;
                    if (person->fitness_value > distance_and_partner.second->fitness_value){
                        smaller_fitness_partner = distance_and_partner.second;
                        bigger_fitness_partner = person;
                    } else {
                        smaller_fitness_partner = person;
                        bigger_fitness_partner = distance_and_partner.second;
                    }

                    do {
                        smaller_fitness_partner->mutate();
                    } while (smaller_fitness_partner->get_distance(bigger_fitness_partner) < DIVERSITY_THRESHOLD);
                }

                auto two_children = person->make_kids_with(distance_and_partner.second);
                new_population.emplace_back(std::move(two_children.first));
                new_population.emplace_back(std::move(two_children.second));
            }

            printf("delete redundant people\n");
            if (new_population.size() > MAX_AMOUNT){
                new_population.erase(new_population.begin() + MAX_AMOUNT, new_population.end());
            }

            printf("if the population is too small, enhance it\n");
            if (MIN_AMOUNT > new_population.size()){
                printf("sort thresholded people according to their prob in descending manner\n");
                std::sort(thresholded_people.begin(), thresholded_people.end(),
                          [] (const std::shared_ptr<Genome>& a, const std::shared_ptr<Genome>& b) {
                              return a->p > b->p;
                          });

                auto limit = MIN_AMOUNT - new_population.size();
                for (int i=0; i<limit; ++i){
                    thresholded_people[i]->mutate();
                    new_population.emplace_back(std::move(thresholded_people[i]));
                }
            }

            printf("check for repetitions\n");
            if (this->ancestors.size() > MAX_AMOUNT){
                for (auto&& successor : new_population){
                    for (auto&& predecessor : this->ancestors){
                        if (successor->get_number() != predecessor->get_number()){
                            while (successor->get_distance(predecessor) < DIVERSITY_THRESHOLD){
                                successor->mutate();
                            }
                        }
                    }
                }
            }

            people = std::move(new_population);

            fitness_log_file << "\n NEW POPULATION SIZE = " << people.size() << "\n";
            fitness_log_file.flush();
        }

        People people;

    private:
        People ancestors;
        std::string genomes_to_load_path = "../precomputed_genomes.txt";

        float get_random(float low, float high){
            return init_rand(mt) * (high - low) + low;
        }
    };

}



#endif //TEST_GENETIC_ALGORITHM_H

#ifndef TEST_GENETIC_ALGORITHM_H
#define TEST_GENETIC_ALGORITHM_H


#include <ctime>
#include <random>
#include <memory>
#include <vector>
#include <experimental/filesystem>
#include <fstream>

namespace genetic_alg{

    using namespace std::experimental::filesystem;

    const int GENOME_LENGTH = 16 + 4 + 16 + 4 + 16;

    std::random_device rd;
    std::mt19937 mt(rd());
    // TODO fix random
    std::uniform_real_distribution<double> init_rand(0., 1.);
    std::uniform_real_distribution<double> mutate_rand(-1., 1.);
    std::uniform_int_distribution<int> crossingover_dist_lo(2, GENOME_LENGTH / 2 - 1);
    std::uniform_int_distribution<int> crossingover_dist_hi(GENOME_LENGTH / 2 + 1, GENOME_LENGTH - 1);

    class Genome{
    public:

        typedef std::pair<std::shared_ptr<Genome>, std::shared_ptr<Genome>> children;

        static int counter;

        Genome(bool is_random = false) {
            number = ++counter;

            if (is_random){
                for (double &i : data) {
                    i = init_rand(mt);
                }
            }
        }

        int get_number(){
            return number;
        }

        children sex_with (const std::shared_ptr<Genome>& that){
            int positions[4] = {0, crossingover_dist_lo(mt),
                                crossingover_dist_hi(mt), GENOME_LENGTH};

            auto child_1 = std::make_unique<Genome>();
            auto child_2 = std::make_unique<Genome>();

            // crossingover in 2 places
            for (int i=0; i<3; ++i){
                if (init_rand(mt) > 0.5){
                    for (int j=positions[i]; j<positions[i+1]; ++j){
                        child_1->data[j] = this->data[j];
                        child_2->data[j] = that->data[j];
                    }
                } else {
                    for (int j=positions[i]; j<positions[i+1]; ++j){
                        child_1->data[j] = that->data[j];
                        child_2->data[j] = this->data[j];
                    }
                }
            }

            return children(std::move(child_1), std::move(child_2));
        }

        double data[GENOME_LENGTH];

        double get_distance(const Genome& that){
            double distance = 0;

            for (int i=0; i<GENOME_LENGTH; ++i){
                distance += (this->data[i] = that.data[i]) * (this->data[i] = that.data[i]);
            }

            return sqrt(distance);
        }

        void mutate(){
            double threshold = init_rand(mt);

            for (double &i : this->data) {
                if (init_rand(mt) > threshold){
                    i += mutate_rand(mt);
                }
            }
        }

        void count_fitness(){
            double iou_sum = 0;
            long iou_counter = 0;
            double current_iou = 0;
            int first_10_counter = 0;

            long fail_counter = 0;

            std::string str;

            for (auto& file_path: directory_iterator(path_to_bboxes_dir / std::to_string(number))){
                std::fstream boxes_file(file_path.path().string());

                while (std::getline(boxes_file, str)){
                    current_iou = std::stod(str);

                    if (current_iou == 0){
                        ++fail_counter;
                    } else if (current_iou == 1.){
                        first_10_counter = 1;
                    } else if (first_10_counter < 10){
                        ++first_10_counter;
                    } else {
                        ++iou_counter;
                        iou_sum += current_iou;
                    }
                }
            }

            robustness = fail_counter;
            accuracy = iou_sum / double(iou_counter);
            fitness_value = robustness + accuracy;
        }

        double count_F_i(double standart_derivation, double mean){
            F_i = 1 + (fitness_value - mean)/(2 * standart_derivation);
            return F_i;
        }

        void count_probability(double F_i_sum){
            p = F_i / F_i_sum;
        }

        double fitness_value;
        double p;

    private:
        int number;
        double F_i;

        double robustness;
        double accuracy;

        std::string path_to_bboxes_dir = "/home/ksenia/bboxes_info";
    };

    int Genome::counter = 0;



    const int START_AMOUNT = 100;
    const int MAX_AMOUNT = 120;

    typedef std::vector<std::shared_ptr<Genome>> People;

    class Population{
    public:
        Population() {
            std::srand ((unsigned int)(time(nullptr) / 2));

            for (int i=0; i<START_AMOUNT; ++i){
                people.emplace_back(std::make_unique<Genome>());
            }
        }

        std::shared_ptr<Genome> find_partner(const std::shared_ptr<Genome> &person) {
            int index = 0;
            double min_distance = 1'000'000'000;

            for (int i = 0; i < people.size(); ++i) {
                if (person->get_number() != people[i]->get_number()){
                    double distance = people[i].get()->get_distance(person.get());

                    if (min_distance > distance) {
                        min_distance = distance;
                        index = i;
                    }
                }
            }

            return std::move(people[index]);
        }

        void create_new_popuation(){
            double mean = 1. / people.size();
            People people_after_selection;
            for (auto& person : people){
                if (person->p >= get_random(0, mean * 3)){
                    people_after_selection.push_back(person);
                }
            }

            People new_population;
            for (auto& person : people_after_selection){
                auto two_children = person->sex_with(find_partner(person));
                new_population.emplace_back(std::move(two_children.first));
                new_population.emplace_back(std::move(two_children.second));
            }

            if (new_population.size() > MAX_AMOUNT){
                auto delta = double(new_population.size() - MAX_AMOUNT);
                double threshold = double(MAX_AMOUNT) - delta / double(MAX_AMOUNT);

                People people_to_remove;
                for(auto& person : new_population){
                    people_to_remove.emplace_back(std::move(person));
                }
                people_to_remove.clear();
            }

            double threshold = 0.2;
            for (auto& person : new_population){
                if (get_random(0, 1) <= threshold){
                    person->mutate();
                }
            }

            people = std::move(new_population);
        }

        People people;

    private:

        double get_random(double low, double high){
            return init_rand(mt) * (high - low) + low;
        }
    };

}



#endif //TEST_GENETIC_ALGORITHM_H

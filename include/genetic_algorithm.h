#ifndef TEST_GENETIC_ALGORITHM_H
#define TEST_GENETIC_ALGORITHM_H


#include <ctime>
#include <random>
#include <memory>
#include <vector>

namespace genetic_alg{

    const int GENOME_LENGTH = 16 + 4 + 16 + 4 + 16;

    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_real_distribution<double> init_rand(0., 1.);
    std::uniform_real_distribution<double> mutate_rand(-1., 1.);
    std::uniform_int_distribution<int> crossingover_dist_lo(2, GENOME_LENGTH / 2 - 1);
    std::uniform_int_distribution<int> crossingover_dist_hi(GENOME_LENGTH / 2 + 1, GENOME_LENGTH - 1);

    class Genome{
    public:

        typedef std::pair<std::unique_ptr<Genome>, std::unique_ptr<Genome>> children;

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

        children sex_with (const Genome& that){
            int positions[4] = {0, crossingover_dist_lo(mt),
                                crossingover_dist_hi(mt), GENOME_LENGTH};

            auto child_1 = std::make_unique<Genome>();
            auto child_2 = std::make_unique<Genome>();

            for (int i=0; i<3; ++i){
                if (init_rand(mt) > 0.5){
                    for (int j=positions[i]; j<positions[i+1]; ++j){
                        child_1->data[j] = this->data[j];
                        child_2->data[j] = that.data[j];
                    }
                } else {
                    for (int j=positions[i]; j<positions[i+1]; ++j){
                        child_1->data[j] = that.data[j];
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

    private:

        double get_random(double low, double high){
            return init_rand(mt) * (high - low) + low;
        }

        int number;

        double p;
        double fitness_value;

        double robustness;
        double accuracy;
    };

    int Genome::counter = 0;



    const int START_AMOUNT = 100;
    const int MAX_AMOUNT = 120;

    typedef std::vector<std::unique_ptr<Genome>> People;

    class Population{
    public:
        Population() {
            std::srand ((unsigned int)(time(nullptr) / 2));

            for (int i=0; i<START_AMOUNT; ++i){
                people.emplace_back(std::make_unique<Genome>());
            }
        }

        std::unique_ptr<Genome> find_parent(const std::unique_ptr<Genome> person) {
            int index = 0;
            double min_distance = 1'000'000'000;

            for (int i = 0; i < people.size(); ++i) {
                double distance = people[i].get()->get_distance(person.get());

                if (min_distance > distance) {
                    min_distance = distance;
                    index = i;
                }
            }

            return std::move(people[index]);
        }

        People people;
    };

}



#endif //TEST_GENETIC_ALGORITHM_H

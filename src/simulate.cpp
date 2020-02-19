#include "simulate.hpp"


int main(){
    const float mean = 5.0;
    const float standard_deviation = 2.0;

    doStuff(mean, standard_deviation);

    return 0;
}


void doStuff(float mean, float standard_deviation){

    bool use_random_seed = false;
    seed_array_t seed_array;

    if (use_random_seed) {
        seed_array = generateRandomSeedArray();
    }
    else {
        seed_array = {0, 1, 2, 3, 4, 5, 6, 7};
    }
    std::mt19937 generator = randomEngine(seed_array);

    std::normal_distribution<double> distribution(mean, standard_deviation);

    std::array<double, 40> thing;

    for (int i=0; i < 40; ++i) {
        thing[i] = distribution(generator);
        std::cout << thing[i] << std::endl;
    }
}


//!  Generates a pseudo random seed
seed_array_t generateRandomSeedArray(){
  std::array<uint_fast32_t, SEED_LEN> seed_array;

  // generates uniformly distributed random numbers
  std::random_device random_source;

  // fills seed_array with random numbers
  std::generate(seed_array.begin(), seed_array.end(), std::ref(random_source));
  return seed_array;
}


//! Generates random number engine
std::mt19937 randomEngine(seed_array_t seed_array) {
  std::seed_seq seed_seq(seed_array.begin(), seed_array.end());
  return std::mt19937{ seed_seq };
}

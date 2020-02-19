#ifndef simulate_hpp
#define simulate_hpp

// Includes
#include <iostream>
#include <random>
#include <array>
#include <algorithm>
#include <functional>
#include <functional>


// Constants
constexpr std::size_t SEED_LEN = 8;


// Typedefs
typedef std::array<uint_fast32_t, SEED_LEN> seed_array_t;


// Functions
void doStuff(float mean, float standard_deviation);
seed_array_t generateRandomSeedArray();
std::mt19937 randomEngine(seed_array_t seed_array);


#endif

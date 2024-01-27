#include <iostream>
#include <queue>
#include <unordered_set>
#include <vector>
#include <tuple>
#include <functional>
#include <math.h>
#include <mpreal.h>
#include <memory>

#include "fad.hpp"

#ifndef AUTO_GRAD_H_
#define AUTO_GRAD_H_

std::shared_ptr<std::vector<std::shared_ptr<Variable>>> auto_grad(std::shared_ptr<Variable> output, std::vector<std::shared_ptr<Variable>> inputs);

#endif

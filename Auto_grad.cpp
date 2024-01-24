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
#include "Auto_grad.hpp"

template <typename TYPE>
auto enumerate(TYPE& inputs){
  std::vector<std::pair<std::size_t, decltype(*std::begin(inputs))&>> enumerated;
  std::size_t index = 0;
  for(auto& item : inputs){    
    enumerated.emplace_back(index, item);
    //auto& itemと書かないと2つとも同じポインタが格納される. なんで?
    index++;
  }
  
  return enumerated;
}

std::vector<std::shared_ptr<Variable>>& auto_grad(std::shared_ptr<Variable> output, std::vector<std::shared_ptr<Variable>> inputs){
  auto ret = new std::vector<std::shared_ptr<Variable>>;
  
  if(output->genertr == nullptr){
    return *ret;
  }
  
  auto compare = [](std::shared_ptr<Variable> a, std::shared_ptr<Variable> b){
		   return a->order < b->order;
		 };
  output->grad_variable = std::make_shared<Variable>(output->grad);
  std::unordered_set<std::shared_ptr<Variable>> visited;
  std::priority_queue<std::shared_ptr<Variable>, std::vector<std::shared_ptr<Variable>>, decltype(compare)> queue{compare};
  queue.push(output);
  
  int depth = 0;
  while(!queue.empty()){
    auto tmp_variable = queue.top();
    auto function = tmp_variable->genertr.get();
    auto gy = tmp_variable->grad_variable;

    auto& gxs = function->auto_grad(gy);
    queue.pop();
    for(const auto& [i,gx] : enumerate(gxs)){
      auto x = function->inputs[i];
      
      if(gx == nullptr){
	continue;
      }
      //std::size_t id_x = std::hash<decltype(x)>{}(x);
      if(x->genertr != nullptr){
	if(visited.find(x) == visited.end()){
	  queue.push(x);	
	}
      }
      if(visited.find(x) == visited.end()){
	x->grad_variable = gx;
	
	visited.insert(x);
      }
      else{
	x->grad_variable = x->grad_variable + gx;
      }
    }
    delete &gxs;    
  }
  for(auto item : inputs){
    ret->push_back(item->grad_variable);
  }
  return *ret;
}

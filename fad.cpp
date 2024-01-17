#include <iostream>
#include <queue>
#include <unordered_set>
#include <vector>
#include <tuple>
#include <memory>
#include <functional>
#include <math.h>
#include <mpreal.h>
#include <gc_cpp.h>

#include "fad.hpp"

using mpfr::mpreal;

class Function;
class Variable;

template <typename TYPE>

auto enumerate(TYPE& inputs){
  std::vector<std::pair<std::size_t, decltype(*std::begin(inputs))&>> enumerated;
  std::size_t index = 0;
  for(auto& item : inputs){
    //std::cout<< index << ", " << item << std::endl;
    
    enumerated.emplace_back(index, item);
    //auto& itemと書かないと2つとも同じポインタが格納される. なんで?
    //enumerated.emplace_back(index, inputs[index]);
    index++;

    // std::for_each(enumerated.begin(), enumerated.end(), [](decltype(enumerated[0]) x) {
    //std::cout <<"test " <<x.first << ',' << x.second << std::endl;});
  }
  
  return enumerated;
}

// Variableクラスの定義
Variable::Variable(const double input){
  data = mpreal(input);
  grad = mpreal(1);
  order = 0;
  genertr = nullptr;
}

Variable::Variable(const mpreal data):data(data),grad(1.0),order(0),genertr(nullptr){
  
}

Variable::Variable(){
  data = 0;
  grad = 0;
  order = 0;
  genertr = nullptr;
}

Variable::~Variable(){
}

void Variable::set_genertr(std::shared_ptr<Function> gen_func){
  genertr = gen_func; 
}

// Variable& Variable::operator=(const Variable& x){
//   data = x.data;
//   grad = x.grad;
//   genertr = x.genertr;
//   return *this;
// }


void Variable::backward(){
  if(genertr == nullptr){
    return;
  }
  
  auto compare = [](std::shared_ptr<Variable> a, std::shared_ptr<Variable> b){
		   return a->order < b->order;
		 };
  
  std::unordered_set<size_t> visited;
  std::priority_queue<std::shared_ptr<Variable>, std::vector<std::shared_ptr<Variable>>, decltype(compare)> queue{compare};
  queue.push(shared_from_this());

  int depth = 0;

  while(!queue.empty()){
    auto output = queue.top();
    auto function = output->genertr;
    mpreal gy = output->grad;

    auto& gxs = function->backward(gy); //返り値はvector<variable*>
    queue.pop();
    for(const auto& [i,gx] : enumerate(gxs)){
      auto x = function->inputs[i];
      if(gx == nullptr){
	continue;
      }
      std::size_t id_x = std::hash<decltype(x)>{}(x);
      if(x->genertr != nullptr){
	if(visited.find(id_x) == visited.end()){
	  queue.push(x);	
	}
      }
      if(visited.find(id_x) == visited.end()){
	x->grad = gx->data;
	
	visited.insert(id_x);
      }
      else{
	x->grad += gx->data;
      }
    }
    depth++;
  }
}

// Functionクラスの定義



std::shared_ptr<Variable> Function::operator()(std::shared_ptr<Function> self, std::shared_ptr<Variable> input1, std::shared_ptr<Variable> input2 = nullptr){
  inputs = {input1, input2};
  int ord = std::max({input1->order, input2->order});
  const mpreal y = forward();
  output = std::make_shared<Variable>(y);
  output->set_genertr(self);
  output->order=ord+1;
  return output;
}

Function::~Function(){

}


// Addクラスのメソッドの定義

mpreal Add::forward(){
  return inputs[0]->data + inputs[1]->data;
}

std::vector<std::shared_ptr<Variable>>& Add::backward(const mpreal gy){
  auto d1=std::make_shared<Variable>(gy);
  auto d2=std::make_shared<Variable>(gy);
  std::vector<std::shared_ptr<Variable>>* ret =new(GC) std::vector<std::shared_ptr<Variable>>{d1,d2};
  return *ret;
}

// Subクラスのメソッドの定義

mpreal Sub::forward(){
  return inputs[0]->data - inputs[1]->data;
}

std::vector<std::shared_ptr<Variable>>& Sub::backward(const mpreal gy){
  auto d1=std::make_shared<Variable>(-1*gy);
  auto d2=std::make_shared<Variable>(gy);
  std::vector<std::shared_ptr<Variable>>* ret =new(GC) std::vector<std::shared_ptr<Variable>>{d1,d2};
  return *ret;
}

// Mulクラスのメソッドの定義

mpreal Mul::forward(){
  return inputs[0]->data * inputs[1]->data;
}

std::vector<std::shared_ptr<Variable>>& Mul::backward(const mpreal gy){
  auto d1=std::make_shared<Variable>(gy*inputs[1]->data);
  auto d2=std::make_shared<Variable>(gy*inputs[0]->data);
  std::vector<std::shared_ptr<Variable>>* ret =new(GC) std::vector<std::shared_ptr<Variable>>{d1,d2};
  return *ret;
}

// Divクラスのメソッドの定義

mpreal Div::forward(){
  return inputs[0]->data/inputs[1]->data;
}

std::vector<std::shared_ptr<Variable>>& Div::backward(const mpreal gy){
  auto d1=std::make_shared<Variable>(gy/inputs[1]->data);
  auto d2=std::make_shared<Variable>(-1*gy*inputs[0]->data / ((inputs[1]->data)*(inputs[1]->data)));
  std::vector<std::shared_ptr<Variable>>* ret =new(GC) std::vector<std::shared_ptr<Variable>>{d1,d2};
  return *ret;
}

// Sqrtクラスの定義
mpreal Sqrt::forward(){
  return sqrt(inputs[0]->data);
}

std::vector<std::shared_ptr<Variable>>& Sqrt::backward(const mpreal gy){
  auto d1=std::make_shared<Variable>(gy/(2*sqrt(inputs[0]->data)));
  std::vector<std::shared_ptr<Variable>>* ret =new(GC) std::vector<std::shared_ptr<Variable>>{d1,nullptr};
  return *ret;
}

// Expクラスの定義
mpreal Exp::forward(){
  return exp(inputs[0]->data);
}

std::vector<std::shared_ptr<Variable>>& Exp::backward(const mpreal gy){
  auto d1=std::make_shared<Variable>(gy*exp(inputs[0]->data));
  std::vector<std::shared_ptr<Variable>>* ret =new(GC) std::vector<std::shared_ptr<Variable>>{d1,nullptr};
  return *ret;
}

// Logクラスの定義
mpreal Log::forward(){
  return exp(inputs[0]->data);
}

std::vector<std::shared_ptr<Variable>>& Log::backward(const mpreal gy){
  auto d1=std::make_shared<Variable>(gy/inputs[0]->data);
  std::vector<std::shared_ptr<Variable>>* ret =new(GC) std::vector<std::shared_ptr<Variable>>{d1,nullptr};
  return *ret;
}

// Sinクラスの定義
mpreal Sin::forward(){
  return exp(inputs[0]->data);
}

std::vector<std::shared_ptr<Variable>>& Sin::backward(const mpreal gy){
  auto d1=std::make_shared<Variable>(gy*cos(inputs[0]->data));
  std::vector<std::shared_ptr<Variable>>* ret =new(GC) std::vector<std::shared_ptr<Variable>>{d1,nullptr};
  return *ret;
}
// Cosクラスの定義
mpreal Cos::forward(){
  return exp(inputs[0]->data);
}

std::vector<std::shared_ptr<Variable>>& Cos::backward(const mpreal gy){
  auto d1=std::make_shared<Variable>(gy*sin(inputs[0]->data));
  std::vector<std::shared_ptr<Variable>>* ret =new(GC) std::vector<std::shared_ptr<Variable>>{d1,nullptr};
  return *ret;
}

// 演算子オーバーロード

std::shared_ptr<Variable>& operator+(std::shared_ptr<Variable>& op1,std::shared_ptr<Variable>& op2){
  auto add_func =std::make_shared<Add>();
  return *(add_func->operator()(add_func, op1, op2));
}

Variable& operator-(Variable& op1,Variable& op2){
  auto sub_func =std::make_shared<Sub>();
  return *(sub_func->operator()(sub_func, &op1, &op2));
}

Variable& operator*(Variable& op1, Variable& op2){
  auto mul_func =std::make_shared<Mul>();
  return *(mul_func->operator()(mul_func, &op1, &op2));
}

Variable& operator/(Variable& op1,Variable& op2){
  auto div_func =std::make_shared<Div>();
  return *(div_func->operator()(div_func, &op1, &op2));
}

Variable& exp(Variable& op){
  Function* func =std::make_shared<Exp>();
  return *(func->operator()(func, &op));
}

Variable& log(Variable& op){
  Function* func =std::make_shared<Log>();
  return *(func->operator()(func, &op));
}

Variable& sin(Variable& op){
  Function* func =std::make_shared<Sin>();
  return *(func->operator()(func, &op));
}

Variable& cos(Variable& op){
  Function* func =std::make_shared<Cos>();
  return *(func->operator()(func, &op));
}

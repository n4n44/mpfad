#include <iostream>
#include <queue>
#include <unordered_set>
#include <vector>
#include <tuple>
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

// std::vector<std::tuple<std::size_t, Variable*>> enumerate(std::vector<Variable*> vec){
//   std::vector<std::tuple<std::size_t, Variable*>> enumerated;
//   std::size_t index = 0;
//   for(auto item : vec){
//     enumerated.emplace_back(index, item);
//     index++;
//   }
  
//   return enumerated;
// }


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


void Variable::set_genertr(Function *gen_func){
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
  
  auto compare = [](Variable* a, Variable* b){
		   return a->order < b->order;
		 };
  
  std::unordered_set<size_t> visited;
  std::priority_queue<Variable*,std::vector<Variable*>,decltype(compare)> queue{compare};
  queue.push(this);

  int depth = 0;

  while(!queue.empty()){
    Variable* output = queue.top();
    Function* function = output->genertr;
    mpreal gy = output->grad;
    std::vector<Variable*>& gxs = function->backward(gy); //返り値はvector<variable*>
    queue.pop();
    for(const auto& [i,gx] : enumerate(gxs)){
      Variable* x = function->inputs[i];
      if(gx == nullptr){
	continue;
      }
      std::size_t id_x = std::hash<Variable*>{}(x);// hashが衝突したらマズイかも
      if(x->genertr != nullptr){
	if(visited.find(id_x) == visited.end()){
	  queue.push(x);	
	}
      }
      if(visited.find(id_x) == visited.end()){
	std::cout<<"cp1"<<std::endl;
	x->grad = gx->data;
	visited.insert(id_x);
      }
      else{
	std::cout<<"cp2"<<std::endl;
	x->grad += gx->data;
      }
    }
    depth++;
  }
}

// Functionクラスの定義



Variable* Function::operator()(Function *self, Variable* input1, Variable* input2 = nullptr){
  inputs = {input1, input2};
  int ord = std::max({input1->order, input2->order});
  const mpreal y = forward();
  output = new Variable(y);
  output->set_genertr(self);
  output->order=ord+1;
  return output;
}


// Addクラスのメソッドの定義

mpreal Add::forward(){
  return inputs[0]->data + inputs[1]->data;
}

std::vector<Variable*>& Add::backward(const mpreal gy){
  Variable* d1=new Variable(gy);
  Variable* d2=new Variable(gy);
  std::vector<Variable*>* ret = new(GC) std::vector<Variable*>{d1,d2};
  return *ret;
}

// Subクラスのメソッドの定義

mpreal Sub::forward(){
  return inputs[0]->data - inputs[1]->data;
}

std::vector<Variable*>& Sub::backward(const mpreal gy){
  Variable* d1=new Variable(-1*gy);
  Variable* d2=new Variable(gy);
  std::vector<Variable*>* ret =new(GC) std::vector<Variable*>{d1,d2};
  return *ret;
}

// Mulクラスのメソッドの定義

mpreal Mul::forward(){
  return inputs[0]->data * inputs[1]->data;
}

std::vector<Variable*>& Mul::backward(const mpreal gy){
  Variable *d1=new Variable(gy*inputs[1]->data);
  Variable *d2=new Variable(gy*inputs[0]->data);
  std::vector<Variable*>* ret =new(GC) std::vector<Variable*>{d1,d2};
  return *ret;
}

// Divクラスのメソッドの定義

mpreal Div::forward(){
  return inputs[0]->data/inputs[1]->data;
}

std::vector<Variable*>& Div::backward(const mpreal gy){
  Variable *d1=new Variable(gy/inputs[1]->data);
  Variable *d2=new Variable(-1*gy*inputs[0]->data / ((inputs[1]->data)*(inputs[1]->data)));
  std::vector<Variable*>* ret =new(GC) std::vector<Variable*>{d1,d2};
  return *ret;
}

// Sqrtクラスの定義
mpreal Sqrt::forward(){
  return sqrt(inputs[0]->data);
}

std::vector<Variable*>& Sqrt::backward(const mpreal gy){
  Variable *d1=new Variable(gy/(2*sqrt(inputs[0]->data)));
  std::vector<Variable*>* ret =new(GC) std::vector<Variable*>{d1,nullptr};
  return *ret;
}

// Expクラスの定義
mpreal Exp::forward(){
  return exp(inputs[0]->data);
}

std::vector<Variable*>& Exp::backward(const mpreal gy){
  Variable *d1=new Variable(gy*exp(inputs[0]->data));
  std::vector<Variable*>* ret =new(GC) std::vector<Variable*>{d1,nullptr};
  return *ret;
}

// Logクラスの定義
mpreal Log::forward(){
  return exp(inputs[0]->data);
}

std::vector<Variable*>& Log::backward(const mpreal gy){
  Variable *d1=new Variable(gy/inputs[0]->data);
  std::vector<Variable*>* ret =new(GC) std::vector<Variable*>{d1,nullptr};
  return *ret;
}

// Sinクラスの定義
mpreal Sin::forward(){
  return exp(inputs[0]->data);
}

std::vector<Variable*>& Sin::backward(const mpreal gy){
  Variable *d1=new Variable(gy*cos(inputs[0]->data));
  std::vector<Variable*>* ret =new(GC) std::vector<Variable*>{d1,nullptr};
  return *ret;
}
// Cosクラスの定義
mpreal Cos::forward(){
  return exp(inputs[0]->data);
}

std::vector<Variable*>& Cos::backward(const mpreal gy){
  Variable *d1=new Variable(gy*sin(inputs[0]->data));
  std::vector<Variable*>* ret =new(GC) std::vector<Variable*>{d1,nullptr};
  return *ret;
}

// 演算子オーバーロード

Variable& operator+(Variable& op1,Variable& op2){
  Function* add_func =new Add();
  return *(add_func->operator()(add_func, &op1, &op2));
}

Variable& operator-(Variable& op1,Variable& op2){
  Function* sub_func =new Sub();
  return *(sub_func->operator()(sub_func, &op1, &op2));
}

Variable& operator*(Variable& op1, Variable& op2){
  auto mul_func =new Mul();
  return *(mul_func->operator()(mul_func, &op1, &op2));
}

Variable& operator/(Variable& op1,Variable& op2){
  Function* div_func =new Div();
  return *(div_func->operator()(div_func, &op1, &op2));
}

// Variable& operator+(double op1, Variable& op2){
//   Variable* temp = new Variable(op1);
//   return temp+op2;
// }

Variable& exp(Variable& op){
  Function* func =new Exp();
  return *(func->operator()(func, &op));
}

Variable& log(Variable& op){
  Function* func =new Log();
  return *(func->operator()(func, &op));
}

Variable& sin(Variable& op){
  Function* func =new Sin();
  return *(func->operator()(func, &op));
}

Variable& cos(Variable& op){
  Function* func =new Cos();
  return *(func->operator()(func, &op));
}

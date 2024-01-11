#include <iostream>
#include <deque>
#include <unordered_set>
#include <vector>
#include <tuple>
#include <functional>
#include <math.h>
#include "mpreal.h"

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
  genertr = nullptr;
}

Variable::Variable(const mpreal data):data(data),grad(1.0),genertr(nullptr){
  
}

Variable::Variable(){
  data = 0;
  grad = 0;
  genertr = nullptr;
}

Variable::~Variable(){
  delete genertr;
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

  std::unordered_set<size_t> visited;
  std::deque<Function*> queue{genertr};

  while(!queue.empty()){
    Function* function = queue.front();
    Variable* output = function->output;

    mpreal gy = output->grad;
    std::vector<Variable*>& gxs = function->backward(gy); //返り値はvector<variable*>
    //std::cout<<gxs[0]<<std::endl;
    queue.pop_front();

    for(auto [i,gx] : enumerate(gxs)){
      Variable* x = function->inputs[i];
      if(gx == nullptr){
	continue;
      }

      std::size_t id_x = std::hash<Variable*>{}(x);
      if(x->genertr != nullptr){
	std::cout<< "cp1 " <<  x->data << ", "<< x->grad<< std::endl;
	queue.push_back(x->genertr);
      }

      if(visited.find(id_x) == visited.end()){
	std::cout<< "cp2 " << x->data << ", "<< x->grad<< std::endl;
	x->grad = gx->data;
	visited.insert(id_x);
      }
      else{
	std::cout<< "cp3 " << x->data << ", "<< x->grad<< std::endl;
	x->grad += gx->data;
      }
    }
  }
}

// Functionクラスの定義



Variable* Function::operator()(Function *self, Variable* input1, Variable* input2 = nullptr){
  inputs = {input1, input2};
  const mpreal y = forward();
  output = new Variable(y);
  output->set_genertr(self);
  return output;
}

Function::~Function(){

}


// Addクラスのメソッドの定義

mpreal Add::forward(){
  return inputs[0]->data + inputs[1]->data;
}

std::vector<Variable*>& Add::backward(const mpreal gy){
  Variable* d1=new Variable(gy);
  Variable* d2=new Variable(gy);
  std::vector<Variable*>* ret = new std::vector<Variable*>{d1,d2};
  return *ret;
}

// Subクラスのメソッドの定義

mpreal Sub::forward(){
  return inputs[0]->data - inputs[1]->data;
}

std::vector<Variable*>& Sub::backward(const mpreal gy){
  Variable* d1=new Variable(-1*gy);
  Variable* d2=new Variable(gy);
  std::vector<Variable*>* ret =new std::vector<Variable*>{d1,d2};
  return *ret;
}

// Mulクラスのメソッドの定義

mpreal Mul::forward(){
  return inputs[0]->data * inputs[1]->data;
}

std::vector<Variable*>& Mul::backward(const mpreal gy){
  Variable *d1=new Variable(gy*inputs[1]->data);
  Variable *d2=new Variable(gy*inputs[0]->data);
  std::vector<Variable*>* ret =new std::vector<Variable*>{d1,d2};
  return *ret;
}

// Divクラスのメソッドの定義

mpreal Div::forward(){
  return inputs[0]->data/inputs[1]->data;
}

std::vector<Variable*>& Div::backward(const mpreal gy){
  Variable *d1=new Variable(gy/inputs[1]->data);
  Variable *d2=new Variable(-1*gy*inputs[0]->data / ((inputs[1]->data)*(inputs[1]->data)));
  std::vector<Variable*>* ret =new std::vector<Variable*>{d1,d2};
  return *ret;
}

// Sqrtクラスの定義
mpreal Sqrt::forward(){
  return sqrt(inputs[0]->data);
}

std::vector<Variable*>& Sqrt::backward(const mpreal gy){
  Variable *d1=new Variable(gy/(2*sqrt(inputs[0]->data)));
  std::vector<Variable*>* ret =new std::vector<Variable*>{d1,nullptr};
  return *ret;
}

// Expクラスの定義
mpreal Exp::forward(){
  return exp(inputs[0]->data);
}

std::vector<Variable*>& Exp::backward(const mpreal gy){
  Variable *d1=new Variable(gy*exp(inputs[0]->data));
  std::vector<Variable*>* ret =new std::vector<Variable*>{d1,nullptr};
  return *ret;
}

// Logクラスの定義
mpreal Log::forward(){
  return exp(inputs[0]->data);
}

std::vector<Variable*>& Log::backward(const mpreal gy){
  Variable *d1=new Variable(gy/inputs[0]->data);
  std::vector<Variable*>* ret =new std::vector<Variable*>{d1,nullptr};
  return *ret;
}

// Sinクラスの定義
mpreal Sin::forward(){
  return exp(inputs[0]->data);
}

std::vector<Variable*>& Sin::backward(const mpreal gy){
  Variable *d1=new Variable(gy*cos(inputs[0]->data));
  std::vector<Variable*>* ret =new std::vector<Variable*>{d1,nullptr};
  return *ret;
}

// Cosクラスの定義
mpreal Cos::forward(){
  return exp(inputs[0]->data);
}

std::vector<Variable*>& Cos::backward(const mpreal gy){
  Variable *d1=new Variable(gy*sin(inputs[0]->data));
  std::vector<Variable*>* ret =new std::vector<Variable*>{d1,nullptr};
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


// int main(){
//   mpreal::set_default_prec(113);
//   const int digits = 20;
//   std::cout.precision(digits);
//   mpreal x("2.5");
//   mpreal y("1.2");
//   Variable X(x);
//   Variable Y(y);
//   //Variable V1,V2;
//   // Variable* V2 = new Variable();
//   Variable V3;
//   //Sub* sub = new Sub();
//   // V3 = *(sub->operator()(sub,&X,&Y));
//   // V2 = V1*V1;
//   // V3 = V2*X;
  
//   // V3.genertr-->outputがV3のポインターを指すようにしたい.
//   V3 = X-Y;
  
//   //std::vector<Variable*> vec={V2,V3};
//   // for(auto item : vec){
//   //   delete item;
//   // }
//   std::cout << sizeof(Variable) << ", " <<sizeof(Function)<< std::endl;
//   //std::cout << V3.data << std::endl;

//   // V3.backward();
  
//   // std::cout << "x = " << X.data << ", x.grad = " << X.grad << std::endl;
//   // std::cout << "y = " << Y.data << ", y.grad = " << Y.grad << std::endl;
//   //delete V3.genertr;
//   //delete sub;
  
//   //delete sub;
//   return 0;
// }


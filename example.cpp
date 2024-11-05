#include <iostream>
#include <iomanip>
#include <deque>
#include <unordered_set>
#include <vector>
#include <tuple>
#include <functional>
#include <cmath>

#include "mpreal.h"
#include "fad.hpp"
#include "Auto_grad.hpp"


// gcc -std=c++17 example.cpp fad.cpp Auto_grad.cpp -lstdc++ -lmpfr 
using mpfr::mpreal;
using namespace std;

shared_ptr<Variable> rosen_brock(vector<shared_ptr<Variable>>& inputs){
  shared_ptr<Variable> ret = make_shared<Variable>(0);
  shared_ptr<Variable> V1;
  shared_ptr<Variable> V2;
  for (int i = 0; i < inputs.size()-1; i++) {
    V1 = inputs[i]*inputs[i];
    V1 = inputs[i+1] - V1;
    V1 = V1*V1;
    V1 = 100*V1;
    V2 = 1-inputs[i];
    V2 = V2*V2;
    ret = ret+V1;
    ret = ret+V2;
  }
  return ret;
}
shared_ptr<Variable> func(vector<shared_ptr<Variable>>& inputs){
  shared_ptr<Variable> ret = 10.0*inputs[0]*inputs[0]+2.0*inputs[0]*inputs[1]+5.0*inputs[1]*inputs[1];
  return ret;
}
int main(){
  mpreal::set_default_prec(113);
  const int digits = 20;
  int N = 3;
  cout.precision(digits);
  mpreal x("2");
  mpreal y("1");
  mpreal z("3");
  auto X = make_shared<Variable>(x);
  auto Y = make_shared<Variable>(y);
  auto Z = make_shared<Variable>(z);
  vector<shared_ptr<Variable>> inputs = {X, Y, Z};
  //  auto F = rosen_brock(inputs);
  auto F = func(inputs);
  cout << "F(X, Y, Z) = " << F->data << endl;
  F->backward();
  cout << "F->backward()"<<endl;
  cout<<"inputs[i]->grad:"<<endl;
  for(auto item:inputs){
    int i = 0;
    cout  << item->grad << " ";
  }
  cout<<endl;
  cout<<endl;
  zero_grad(inputs);
  
  auto grad = auto_grad(F, inputs);
  cout<<"grad = auto_grad(F, inputs)"<<endl;
  cout<<"grad[i]->data:"<<endl;
  for(auto& item : *grad){
    cout <<setw(5) << item->data <<endl;
  }
  cout << endl;
  vector<shared_ptr<vector<shared_ptr<Variable>>>> hesse;
  cout << "auto_grad(grad[i],inputs) : // hessian matrix" <<endl;
  for(auto item: *grad){
    auto grad_2 = auto_grad(item, inputs);
    for(auto dd : *grad_2){
      cout <<setw(5) <<dd->data;
    }
    hesse.push_back(grad_2);
    cout<<endl;
  }
  
  // cout<< endl;
  // cout << "f(x,y,z) = " << (1-x)*(1-x)+100*(y-x*x)*(y-x*x)+(1-y)*(1-y)+100*(z-y*y)*(z-y*y)<<endl;
  // cout <<"df/dx(x,y,z) = "<< -400*x*(y-x*x)+2*(x-1) << endl;
  // cout <<"df/dy(x,y,z) = "<<-400*y*(z-y*y)+2*(y-1)+200*(y-x*x) << endl;
  // cout <<"df/dz(x,y,z) = " <<200*(z-y*y) << endl;
  return 0;
}

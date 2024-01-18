#include <iostream>
#include <deque>
#include <unordered_set>
#include <vector>
#include <memory>
#include <tuple>
#include <functional>
#include <math.h>
#include <mpreal.h>

#ifndef FAD_H_
#define FAD_H_

using mpfr::mpreal;

class Variable;
class Function;

template <typename TYPE>

auto enumerate(TYPE& inputs);

// std::vector<std::tuple<std::size_t, Variable*>> enumerate(std::vector<Variable*> vec);
// class variable: public Variable{
// public:
//   std::shared_ptr<Variable>;
//   std::unique_ptr<Function>;
//   mpreal grad;
//   void backward();
// }

class Variable: public std::enable_shared_from_this<Variable>{
public:
  mpreal data;
  mpreal grad;
  std::unique_ptr<Function> genertr;
  int order;
  Variable(const double data);
  Variable(const mpreal data);
  Variable();

  ~Variable();
  
  void set_genertr(std::unique_ptr<Function>& gen_func);
  void backward();
  //Variable& operator=(const Variable& x);
};

class Function: public std::enable_shared_from_this<Function>{
public:
  std::vector<std::shared_ptr<Variable>> inputs;
  std::shared_ptr<Variable> output;
  std::shared_ptr<Variable> operator()(std::unique_ptr<Function>& self,std::shared_ptr<Variable> input1, std::shared_ptr<Variable> input2);

  ~Function();

  virtual mpreal forward()=0;
  virtual std::vector<std::shared_ptr<Variable>>& backward(const mpreal gy)=0;
};

class Add : public Function{
  mpreal forward();

  std::vector<std::shared_ptr<Variable>>& backward(const mpreal gy);
};

class Mul : public Function{
  mpreal forward();

  std::vector<std::shared_ptr<Variable>>& backward(const mpreal gy);
};

class Sub : public Function{
  mpreal forward();

  std::vector<std::shared_ptr<Variable>>& backward(const mpreal gy);
};

class Div : public Function{
  mpreal forward();

  std::vector<std::shared_ptr<Variable>>& backward(const mpreal gy);
};

class Sqrt : public Function{
  mpreal forward();

  std::vector<std::shared_ptr<Variable>>& backward(const mpreal gy);
};

class Exp : public Function{
  mpreal forward();

  std::vector<std::shared_ptr<Variable>>& backward(const mpreal gy);  
};

class Log : public Function{
  mpreal forward();

  std::vector<std::shared_ptr<Variable>>& backward(const mpreal gy);
};

class Sin : public Function{
  mpreal forward();

  std::vector<std::shared_ptr<Variable>>& backward(const mpreal gy);  
};

class Cos : public Function{
  mpreal forward();

  std::vector<std::shared_ptr<Variable>>& backward(const mpreal gy);
};

std::shared_ptr<Variable> operator+(std::shared_ptr<Variable>& lhs, std::shared_ptr<Variable>& rhs);

std::shared_ptr<Variable> operator-(std::shared_ptr<Variable>& lhs, std::shared_ptr<Variable>& rhs);

std::shared_ptr<Variable> operator*(std::shared_ptr<Variable>& lhs, std::shared_ptr<Variable>& rhs);

std::shared_ptr<Variable> operator/(std::shared_ptr<Variable>& lhs, std::shared_ptr<Variable>& rhs);

std::shared_ptr<Variable> sqrt(std::shared_ptr<Variable>& op);

std::shared_ptr<Variable> exp(std::shared_ptr<Variable>& op);

std::shared_ptr<Variable> sin(std::shared_ptr<Variable>& op);

std::shared_ptr<Variable> log(std::shared_ptr<Variable>& op);

std::shared_ptr<Variable> cos(std::shared_ptr<Variable>& op);

#endif


  

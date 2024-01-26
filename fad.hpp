#include <iostream>
#include <queue>
#include <unordered_set>
#include <vector>
#include <memory>
#include <tuple>
#include <functional>
#include <math.h>
#include <mpreal.h>
#include <gc_cpp.h>

#ifndef FAD_H_
#define FAD_H_

using mpfr::mpreal;

class Variable;
class Function;

template <typename TYPE>
auto enumerate(TYPE& inputs);

class Variable: public std::enable_shared_from_this<Variable>{
public:
  mpreal data;
  mpreal grad;
  std::unique_ptr<Function> genertr;
  int order;
  std::weak_ptr<Variable> grad_variable;
  Variable(const double data);
  Variable(const mpreal data);
  Variable();
  
  void set_genertr(std::unique_ptr<Function>& gen_func);
  void backward();
};


class Function: public std::enable_shared_from_this<Function>{
public:
  std::vector<std::shared_ptr<Variable>> inputs;
  std::weak_ptr<Variable> output;
  std::shared_ptr<Variable> operator()(std::unique_ptr<Function>& self,std::shared_ptr<Variable> input1, std::shared_ptr<Variable> input2);

  virtual mpreal forward()=0;
  virtual std::vector<std::shared_ptr<Variable>>& backward(const mpreal gy)=0;
  virtual std::vector<std::shared_ptr<Variable>>& auto_grad(const std::shared_ptr<Variable>& gy)=0;
};

class Add : public Function{
  mpreal forward();

  std::vector<std::shared_ptr<Variable>>& backward(const mpreal gy);

  std::vector<std::shared_ptr<Variable>>& auto_grad(const std::shared_ptr<Variable>& gy);
};

class Mul : public Function{
  mpreal forward();

  std::vector<std::shared_ptr<Variable>>& backward(const mpreal gy);
  
  std::vector<std::shared_ptr<Variable>>& auto_grad(const std::shared_ptr<Variable>& output);
};

class Sub : public Function{
  mpreal forward();

  std::vector<std::shared_ptr<Variable>>& backward(const mpreal gy);

  std::vector<std::shared_ptr<Variable>>& auto_grad(const std::shared_ptr<Variable>& gy);
};

class Div : public Function{
  mpreal forward();

  std::vector<std::shared_ptr<Variable>>& backward(const mpreal gy);

  std::vector<std::shared_ptr<Variable>>& auto_grad(const std::shared_ptr<Variable>& gy);
};

class Sqrt : public Function{
  mpreal forward();

  std::vector<std::shared_ptr<Variable>>& backward(const mpreal gy);

  std::vector<std::shared_ptr<Variable>>& auto_grad(const std::shared_ptr<Variable>& gy);
};

class Exp : public Function{
  mpreal forward();

  std::vector<std::shared_ptr<Variable>>& backward(const mpreal gy);

  std::vector<std::shared_ptr<Variable>>& auto_grad(const std::shared_ptr<Variable>& gy);
};

class Log : public Function{
  mpreal forward();

  std::vector<std::shared_ptr<Variable>>& backward(const mpreal gy);

  std::vector<std::shared_ptr<Variable>>& auto_grad(const std::shared_ptr<Variable>& gy);
};

class Sin : public Function{
  mpreal forward();

  std::vector<std::shared_ptr<Variable>>& backward(const mpreal gy);  

  std::vector<std::shared_ptr<Variable>>& auto_grad(const std::shared_ptr<Variable>& gy);
};

class Cos : public Function{
  mpreal forward();

  std::vector<std::shared_ptr<Variable>>& backward(const mpreal gy);

  std::vector<std::shared_ptr<Variable>>& auto_grad(const std::shared_ptr<Variable>& gy);
};

std::shared_ptr<Variable> operator+(const std::shared_ptr<Variable>& lhs, const std::shared_ptr<Variable>& rhs);

std::shared_ptr<Variable> operator-(const std::shared_ptr<Variable>& lhs, const std::shared_ptr<Variable>& rhs);

std::shared_ptr<Variable> operator*(const std::shared_ptr<Variable>& lhs, const std::shared_ptr<Variable>& rhs);

std::shared_ptr<Variable> operator/(const std::shared_ptr<Variable>& lhs, const std::shared_ptr<Variable>& rhs);

std::shared_ptr<Variable> operator+(const mpreal& lhs, const std::shared_ptr<Variable>& rhs);

std::shared_ptr<Variable> operator-(const mpreal& lhs, const std::shared_ptr<Variable>& rhs);

std::shared_ptr<Variable> operator*(const mpreal& lhs, const std::shared_ptr<Variable>& rhs);

std::shared_ptr<Variable> operator/(const mpreal& lhs, const std::shared_ptr<Variable>& rhs);

std::shared_ptr<Variable> operator+(const double& lhs, const std::shared_ptr<Variable>& rhs);

std::shared_ptr<Variable> operator-(const double& lhs, const std::shared_ptr<Variable>& rhs);

std::shared_ptr<Variable> operator*(const double& lhs, const std::shared_ptr<Variable>& rhs);

std::shared_ptr<Variable> operator/(const double& lhs, const std::shared_ptr<Variable>& rhs);

std::shared_ptr<Variable> sqrt(const std::shared_ptr<Variable>& op);

std::shared_ptr<Variable> exp(const std::shared_ptr<Variable>& op);

std::shared_ptr<Variable> sin(const std::shared_ptr<Variable>& op);

std::shared_ptr<Variable> log(const std::shared_ptr<Variable>& op);

std::shared_ptr<Variable> cos(const std::shared_ptr<Variable>& op);

void zero_grad(std::vector<std::shared_ptr<Variable>>& inputs);

#endif

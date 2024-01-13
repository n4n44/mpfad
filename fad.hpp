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

class Variable{
public:
  mpreal data;
  mpreal grad;
  Function* genertr;
  int order;
  Variable(const double data);
  Variable(const mpreal data);
  Variable();

  ~Variable();
  
  void set_genertr(Function *gen_func);
  void backward();
  //Variable& operator=(const Variable& x);
};

class Function{
public:
  std::vector<Variable*> inputs;
  Variable* output;
  Variable* operator()(Function *self, Variable* input1, Variable* input2);

  ~Function();

  virtual mpreal forward()=0;
  virtual std::vector<Variable *>& backward(const mpreal gy)=0;
};

class Add : public Function{
  mpreal forward();

  std::vector<Variable*>& backward(const mpreal gy);
};

class Mul : public Function{
  mpreal forward();

  std::vector<Variable*>& backward(const mpreal gy);
};

class Sub : public Function{
  mpreal forward();

  std::vector<Variable*>& backward(const mpreal gy);
};

class Div : public Function{
  mpreal forward();

  std::vector<Variable*>& backward(const mpreal gy);
};

class Sqrt : public Function{
  mpreal forward();

  std::vector<Variable*>& backward(const mpreal gy);
};

class Exp : public Function{
  mpreal forward();

  std::vector<Variable*>& backward(const mpreal gy);  
};

class Log : public Function{
  mpreal forward();

  std::vector<Variable*>& backward(const mpreal gy);
};

class Sin : public Function{
  mpreal forward();

  std::vector<Variable*>& backward(const mpreal gy);  
};

class Cos : public Function{
  mpreal forward();

  std::vector<Variable*>& backward(const mpreal gy);
};

Variable& operator+(Variable& op1,Variable& op2);

Variable& operator-(Variable& op1,Variable& op2);

Variable& operator*(Variable& op1,Variable& op2);

Variable& operator/(Variable& op1,Variable& op2);

Variable& sqrt(Variable& op);

Variable& exp(Variable& op);

Variable& sin(Variable& op);

Variable& cos(Variable& op);

Variable& log(Variable& op);

#endif


  

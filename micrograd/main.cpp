#include "value.h"

#include <iostream>

int main() {
  Value<double> w1(-3.0, "w1");
  Value<double> x1(2.0, "x1");

  Value<double> w2(1.0, "w2");
  Value<double> x2(0.0, "x2");

  Value<double> b(6.8813735870195432, "b");


  Value<double> w1x1 = w1 * x1;
  w1x1.SetName("w1x1");
  Value<double> w2x2 = w2 * x2;
  w2x2.SetName("w2x2");


  Value<double> w1x1w2x2 = w1x1 + w2x2;
  w1x1w2x2.SetName("w1x1w2x2");


  Value<double> n = w1x1w2x2 + b;
  n.SetName("n");


  Value<double> o = n.Tanh();
  o.SetName("o");

  o.Backward();

  return 0;
}

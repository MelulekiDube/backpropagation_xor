#ifndef MY_SYMBOL_H
#define MY_SYMBOL_H

#include <vector>

class Neuron{
double input;
double error;
double output;
double target;

std::vector<double> weights;

public:
  std::vector<double> inputs;
  Neuron(){};
  Neuron(std::vector<double> _weights):weights(std::move(_weights)){};

  double getError() const {
         return error;
     }

     void setError(double error) {
         Neuron::error = error;
     }

     const std::vector<double> &getWeights() const {
         return weights;
     }

     void setWeights(const std::vector<double> &weights) {
         Neuron::weights = weights;
     }

     double getOutput() const {
         return output;
     }

     void setOutput(double output) {
         Neuron::output = output;
     }

     double getInput() const {
         return input;
     }

     void setInput(double input) {
         Neuron::input = input;
 }

 double getTarget() const {
     return target;
 }

 void setTarget(double target) {
     Neuron::target = target;
}

};

#endif /* MY_SYMBOL_H */

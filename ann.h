#ifndef MY_SYMBOL_H
#define MY_SYMBOL_H

class Neuron{
double input;
double error;
double output;
std::vector<double> inputs;
std::vector<double> weights;

public:
  Neuron(){};

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

};

#endif /* MY_SYMBOL_H */

#include "ann.h"
#include <cmath>
#include <iostream>
#include <cstdlib>
#include <ctime>
using namespace std;

double transferFunction(double x);
vector<vector<Neuron> > layers;
void forward_pass(vector<vector<Neuron> > &layers);
void back_pass(vector<vector<Neuron> > &layers);
void initialiseLayers(vector<vector<Neuron> > &layers,vector<double> inputs,double target,int numberOfHiddenNeurons);
double getRandomWeight();
double getError(double output,double target);
double calcErrorOutput(double output,double target);

int main() {

vector<vector<double> > initialInputs ={{0,0,0},{0,0,1},{0,1,0},{0,1,1},{1,0,0},{1,0,1},{1,1,0},{1,1,1}};
std::vector<double> targets={0,1,1,1,1,1,1,0};
int index =0;
int numberOfHiddenNeurons=2;
initialiseLayers( layers, initialInputs[index],targets[index],numberOfHiddenNeurons);
for(unsigned int i=0;i<20000;++i){
forward_pass(layers);
back_pass(layers);
//cout<<"test:\nsizel1: "<<layers[1].size()<<" sizew2 "<<layers[2][0].getWeights().size()<<endl;
cout<<i<<" The output is: "<<layers[2][0].getOutput()<<endl;
//cout<<"input length "<<layers[1][0].inputs.size()<<endl;
}
return 0;
}






void back_pass(vector<vector<Neuron> > &layers){

    /*double h1_error= layers[1][0].getOutput()*(1-layers[1][0].getOutput())*(layers[2][0].getWeights()[0]*layers[2][0].getError());

    double h2_error= layers[1][1].getOutput()*(1-layers[1][1].getOutput())*(layers[2][0].getWeights()[1]*layers[2][0].getError());
*/

//calculate error for hidden nodes
for(unsigned int i=0;i<layers[1].size();++i){
    layers[1][i].setError(layers[1][0].getOutput()*(1-layers[1][0].getOutput())*(layers[2][0].getWeights()[i]*layers[2][0].getError()));

}


  //  layers[1][0].setError(h1_error);
  //  layers[1][1].setError(h2_error);
   //  cout<<"h1 error "<<h1_error<<endl;
   // cout<<"h2 error "<<h2_error<<endl;



  /*  double v11=layers[1][0].getWeights()[0]+(0.1*layers[1][0].getError()*layers[1][0].inputs[0]);
    double v12=layers[1][0].getWeights()[1]+(0.1*layers[1][0].getError()*layers[1][0].inputs[1]);
    double v13=layers[1][0].getWeights()[2]+(0.1*layers[1][0].getError()*layers[1][0].inputs[2]);

    vector<double> weights_v ={v11,v12,v13};
    layers[1][0].setWeights(weights_v);

    double v21=layers[1][1].getWeights()[0]+(0.1*layers[1][1].getError()*layers[1][1].inputs[0]);
    double v22=layers[1][1].getWeights()[1]+(0.1*layers[1][1].getError()*layers[1][1].inputs[1]);
    double v23=layers[1][1].getWeights()[2]+(0.1*layers[1][1].getError()*layers[1][1].inputs[2]);

    vector<double> weights_v1 ={v21,v22,v23};
    */

for(unsigned int i=0;i<layers[1].size();++i){
      vector<double> weights_v1;
    for(unsigned int j=0;j<layers[1][i].inputs.size();++j){
      weights_v1.push_back(layers[1][i].getWeights()[j]+(0.1*layers[1][i].getError()*layers[1][i].inputs[j]));
    }

    layers[1][i].setWeights(weights_v1);

}





      vector<double> weights2 ;
    for(unsigned int w=0;w<layers[2][0].inputs.size();++w){
        weights2.push_back(layers[2][0].getWeights()[w]+(0.1*layers[2][0].getError()*layers[2][0].inputs[w]));
    }



    layers[2][0].setWeights(weights2);

    }


/**
 * passes the data to the next layer in the network
 * @param layers
 */

void forward_pass(vector<vector<Neuron> > &layers){
//  cout<<layers.size()<<endl;
    vector<Neuron> prev_layer;
    for(unsigned int i=1;i<layers.size();++i){
        prev_layer=layers[i-1];
        for(unsigned int j=0;j<layers[i].size();++j){

            double input=0;

  vector<double> inputs;
  for (unsigned int p=0;p<prev_layer.size();++p){
    inputs.push_back(prev_layer[p].getOutput());//{,prev_layer[1].getOutput()};
    input+=prev_layer[p].getOutput()*layers[i][j].getWeights()[0];

  }

            layers[i][j].inputs=inputs;
            layers[i][j].setInput(input);
            layers[i][j].setOutput(transferFunction(input));

        }
    }
    //set the error for the output Neuron
    layers[2][0].setError(calcErrorOutput(  layers[2][0].getOutput(),  layers[2][0].getTarget()));
  //  cout<<"The error is: "<<layers[2][0].getError()<<endl;

}

/**
 * initialise the layers with inputs and random weights
 * @param layers neural network layers
 * @param inputs initial inputs
 */
void initialiseLayers(vector<vector<Neuron> > &layers,vector<double> inputs,double target,int numberOfHiddenNeurons){

layers.reserve(3);
  //initialise the first layer, set inputs
  std::vector<Neuron> temp;
  std::vector<Neuron> temp1;


for(int i=0;i<3;++i){
Neuron l1;

l1.setInput(inputs[i]);
l1.setOutput(inputs[i]);
temp.push_back(l1);
}

layers.push_back(temp);

// initialise the hidden and output layer with random weights

for (int j=0;j<numberOfHiddenNeurons;++j){
  std::vector<double> weights2={getRandomWeight(),getRandomWeight(),getRandomWeight()};
  Neuron l2(weights2);
  temp1.push_back(l2);
}
layers.push_back(temp1);

//initialise the last Neuron with random weights
std::vector<double> weights3={getRandomWeight(),getRandomWeight()};

Neuron l3(weights3);
l3.setTarget(target);
std::vector<Neuron> temp2={l3};//temp vector prevents seg_core dump error
layers.push_back(temp2);
// cout<<layers.size()<<endl;


}



double transferFunction(double x) {

    return 1/(1+exp(-x));
}

double calcErrorOutput(double output,double target){

 return output*(1-output)*(target-output);
}

double getRandomWeight(){
//  srand(time(0));
  double r = ((double) rand() / (RAND_MAX));
//cout<<"random is "<<r<<endl;
  return r;
}

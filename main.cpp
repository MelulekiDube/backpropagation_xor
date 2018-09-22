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
void initialiseLayers(vector<vector<Neuron> > &layers,vector<double> inputs,double target);
double getRandomWeight();
double getError(double output,double target);
double calcErrorOutput(double output,double target);

int main() {
cout<<"hello world"<<endl;
vector<vector<double> > initialInputs ={{0,0,0},{0,0,1},{0,1,0},{0,1,1},{1,0,0},{1,0,1},{1,1,0},{1,1,1}};
std::vector<double> targets={0,1,1,1,1,1,1,0};
int index =2;
initialiseLayers( layers, initialInputs[index],targets[index]);
for(int i=0;i<10;++i){
forward_pass(layers);
back_pass(layers);
cout<<"The output is: "<<layers[2][0].getOutput()<<endl;
//cout<<"input length "<<layers[1][0].inputs.size()<<endl;
}
return 0;
}






void back_pass(vector<vector<Neuron> > &layers){

    double h1_error= layers[1][0].getOutput()*(1-layers[1][0].getOutput())*(layers[2][0].getWeights()[0]*layers[2][0].getError());

    double h2_error= layers[1][1].getOutput()*(1-layers[1][1].getOutput())*(layers[2][0].getWeights()[1]*layers[2][0].getError());
    layers[1][0].setError(h1_error);
    layers[1][1].setError(h2_error);
    //cout<<"h1 error "<<h1_error<<endl;
  //  cout<<"h1 error "<<h2_error<<endl;



    double v11=layers[1][0].getWeights()[0]+(0.1*layers[1][0].getError()*layers[1][0].inputs[0]);
    double v12=layers[1][0].getWeights()[1]+(0.1*layers[1][0].getError()*layers[1][0].inputs[1]);
    double v13=layers[1][0].getWeights()[2]+(0.1*layers[1][0].getError()*layers[1][0].inputs[2]);

    vector<double> weights_v ={v11,v12,v13};
    layers[1][0].setWeights(weights_v);

    double v21=layers[1][1].getWeights()[0]+(0.1*layers[1][1].getError()*layers[1][1].inputs[0]);
    double v22=layers[1][1].getWeights()[1]+(0.1*layers[1][1].getError()*layers[1][1].inputs[1]);
    double v23=layers[1][1].getWeights()[2]+(0.1*layers[1][1].getError()*layers[1][1].inputs[2]);

    vector<double> weights_v1 ={v21,v22,v23};
    layers[1][1].setWeights(weights_v1);

//    cout<<"v11: "<<v11<<endl;
//cout<<"v11: "<<v12<<endl;
//cout<<"v11: "<<v21<<endl;
//cout<<"v11: "<<v22<<endl;


    double w11=layers[2][0].getWeights()[0]+(0.1*layers[2][0].getError()*layers[2][0].inputs[0]);
  cout<<"weight w11: "<<w11<<" weight1 "<<layers[2][0].getWeights()[0]<<endl;

    double w12=layers[2][0].getWeights()[1]+(0.1*layers[2][0].getError()*layers[2][0].inputs[1]);
   cout<<"weight w12: "<<w12<<" weight1 "<<layers[2][0].getWeights()[1]<<endl;
    vector<double> weights ={w11,w12};
    layers[1][0].setWeights(weights);

  /*double w21=layers[2][1].getWeights()[0]+(0.1*layers[2][1].getError()*layers[2][1].inputs[0]);
   // cout<<"weight w21: "<<w21<<endl;
    double w22=layers[2][1].getWeights()[1]+(0.1*layers[2][1].getError()*layers[2][1].inputs[1]);
    //cout<<"weight w22: "<<w22<<endl;
    vector<double> weights1 ={w21,w22};
    layers[2][1].setWeights(weights1);
*/
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

            double input= prev_layer[0].getOutput()*layers[i][j].getWeights()[0]+prev_layer[1].getOutput()*layers[i][j].getWeights()[1];
            //cout<<"outputs "<<prev_layer[0].getOutput()<<" "<<prev_layer[1].getOutput()<<endl;
        //   cout<<"out"<<i<<" "<<j<<": "<<transferFunction(input)<<" input "<<input<<endl;
  vector<double> inputs;
          if(i==2){
           inputs={prev_layer[0].getOutput(),prev_layer[1].getOutput()};
          }
          else{

               inputs={prev_layer[0].getOutput(),prev_layer[1].getOutput(),prev_layer[2].getOutput()};
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
void initialiseLayers(vector<vector<Neuron> > &layers,vector<double> inputs,double target){

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
int numberOfHidden =2;
for (int j=0;j<numberOfHidden;++j){
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

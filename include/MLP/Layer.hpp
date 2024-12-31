#ifndef NEW_LAYER_HPP
#define NEW_LAYER_HPP 

#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include "MLP/Modifiers.hpp"


#define WEIGHT_DECAY 1e-7

namespace E=Eigen;
using E::MatrixXf;
using E::VectorXf;
using E::VectorXi;
using MatFunction = std::function<MatrixXf(const MatrixXf&)>;

class Layer{
private:
  std::string store_path;
  // Main components
  MatrixXf weights;
  VectorXf biases;
  MatrixXf activations;
  MatrixXf errors;
  // Non-lineariy
  MatFunction f,f_dot;
  // Changable parameters
  int batch_size;
  float rate;

  // MLP modifiers
   
  // Adam for each layer
  Adam adam; 
  Dropout dropout;

public:
  // Constructors

  Layer(){};

  Layer(int input_size,int output_size,int batch_size,float rate);

  Layer(std::string folder_path,const int batch_size);

  // References for weights and errors
  const MatrixXf& wRef(){return weights;}
  const MatrixXf& eRef(){return errors;}

  // Config
  
  void setStorePath(std::string store_path){this->store_path=store_path;}
  void setRate(float rate){this->rate=rate;}
  void setBatchSize(int batch_size);

  // Initializes the weights to random small values.
  void assertRandomWeights();
  // He Initialization accoutning for fan-in 
  void HeRandomInit();

  void setAdam(float rate,float beta_1,float beta_2,int batch_size){
    this->adam=Adam(rate,beta_1,beta_2,this->weights,this->biases,batch_size);
  }

  // Simple prints
  void printWeights();
  void printBiases();
  void printActivations();
  void printErrors();



  // Method with and without dropout
  void activateDropout(const MatrixXf& input);
  void activateNormal(const MatrixXf& input);
  // Softmax output
  void softMaxForward(const MatrixXf& input);
  void softMaxBackward(const VectorXi& correct_labels);
  // Back propagation
  void activateErrors(const MatrixXf& next_weights,
                      const MatrixXf& next_errors);

  // Standard update
  void updateWeights(const MatrixXf& input);

  // I/O to disk 
  void store();
  void load();
};

#endif

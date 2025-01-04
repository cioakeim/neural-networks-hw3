#ifndef NEW_MLP_HPP
#define NEW_MLP_HPP

#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include <memory>
#include "MLP/Layer.hpp"
#include "MLP/BaseLayer.hpp"
#include "MLP/Modifiers.hpp"
#include "CommonLib/basicStructs.hpp"

#define WEIGHT_DECAY 1e-7

namespace E=Eigen;
using E::MatrixXf;
using E::VectorXf;
using E::VectorXi;
using MatFunction = std::function<MatrixXf(const MatrixXf&)>;



class MLP{
protected:
  // For I/O purposes
  std::string store_path;
  // Structure
  std::vector<std::shared_ptr<BaseLayer>> layers;
  std::vector<std::shared_ptr<LayerInterface>> interfaces;
  // Training set
  SampleMatrix& training_set;
  SampleMatrix& test_set;
  int batch_size;


  
public:

  MLP(SampleMatrix& training_set,
      SampleMatrix& test_set,
      int batch_size):
  training_set(training_set),
  test_set(test_set),
  batch_size(batch_size){};


  void addLayer(LayerProperties properties);
  void addInterface(std::shared_ptr<LayerInterface> interface);

  void setStorePath(std::string path);
  void setBatchSize(int batch_size){this->batch_size=batch_size;}

  void forward(const PassContext& context);
  void backward(const PassContext& context);

  float runEpoch();


  // Test the epoch result (return the loss function and accuracy)
  void testModel(float& J_test,float& accuracy);

  // Store to place
  void store();
  void load();

  // Config:
  void init();

  
};





#endif

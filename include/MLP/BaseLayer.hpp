#ifndef BASE_LAYER_HPP
#define BASE_LAYER_HPP

#include <memory>
#include <iostream>
#include <string>
#include <Eigen/Dense>
#include "MLP/Optimizer.hpp"
#include "MLP/LayerConfigs.hpp"

namespace E=Eigen;
using E::MatrixXf;
using E::VectorXf;
using E::VectorXi;
using MatFunction = std::function<MatrixXf(const MatrixXf&)>;




/**
 * @brief Virtual class of layer object.
 */
class BaseLayer{
protected:
  std::string store_path; //< Where the data is stored
  // For interfacing with other layers
  std::shared_ptr<LayerInterface> input_interface=nullptr;
  std::shared_ptr<LayerInterface> output_interface=nullptr;
  // For locking the parameters
  bool lockParams=false;

public:
  ~BaseLayer()=default;
  // Config neighboring layers
  void setStorePath(std::string path){this->store_path=path;}
  // Interface setters/getters 
  void setInputInterface(std::shared_ptr<LayerInterface> input_interface){
    this->input_interface=input_interface;
  }
  void setOutputInterface(std::shared_ptr<LayerInterface> output_interface){
    this->output_interface=output_interface;
  }
  // Locking 
  void lock(){lockParams=false;}
  void unlock(){lockParams=false;}

  std::shared_ptr<LayerInterface> getInputInterface(){return input_interface;}
  std::shared_ptr<LayerInterface> getOutputInterface(){return output_interface;}


  // Initialization
  virtual void configure(LayerConfig config)=0;
  virtual void init()=0; 

  // Forward
  virtual void forward(const PassContext& context)=0;

  virtual float loss(const PassContext& context)=0;
  virtual int prediction_success(const PassContext& context)=0;

  // Backward
  virtual void backward(const PassContext& context)=0;;

  // I/O
  virtual void store()=0;
  virtual void load()=0;

  // For debugging
  virtual void printStateInfo()=0;
};

#endif

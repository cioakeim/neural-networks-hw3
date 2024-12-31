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
 * @brief Context of a pass
 */
struct PassContext{
  const MatrixXf input;
  const VectorXi labels;
};


/**
 * @brief Virtual class of layer object.
 */
class BaseLayer{
protected:
  std::string store_path; //< Where the data is stored
  std::shared_ptr<BaseLayer> previous=nullptr;
  std::shared_ptr<BaseLayer> next;
  // For interfacing with other layers
  InterfaceParams input_param,output_param;
  MatrixXf input_error; //< Signal is ready for next node
  MatrixXf output;
  // All layers have 1 optimizer
  // All layers have a non-linearity
  MatFunction f,f_dot;

public:
  ~BaseLayer()=default;
  // Config neighboring layers
  void setStorePath(std::string path){this->store_path=path;}
  void setPreviousLayer(std::shared_ptr<BaseLayer> prev){this->previous=prev;
    std::cout<<prev->outputRef().rows()<<std::endl;
  }

  void setNextLayer(std::shared_ptr<BaseLayer> next){this->next=next;
  }
  void setNonLinearity(MatFunction f,MatFunction f_dot){
    this->f=f;this->f_dot=f_dot;}
  MatFunction getFDot(){return f_dot;}
  // Get type

  // For interfacing with the rest base layers
  const MatrixXf& outputRef(){return output;};
  const MatrixXf& inputErrorRef(){return input_error;};
  InterfaceParams& outputParamRef(){return output_param;}
  InterfaceParams& inputParamRef(){return output_param;}

  // Initialization
  virtual void configure(LayerConfig config)=0;
  virtual void init()=0; 


  // Forward
  virtual void forward(const PassContext& context)=0;

  virtual float loss(const PassContext& context)=0;
  virtual int prediction_success(const PassContext& context)=0;

  // Backward (same idea as above)
  virtual void backward(const PassContext& context)=0;;

  // I/O
  virtual void store()=0;
  virtual void load()=0;
};

#endif

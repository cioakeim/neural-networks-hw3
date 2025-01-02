#ifndef AUTOENCODER_HPP
#define AUTOENCODER_HPP

#include "MLP/MLP.hpp"

/**
 * @brief Autoencoder class 
  */
class AutoEncoder:MLP{
protected:
  // Stack structure for layers: enc_stack[0]==in / dec_stack[0]==out
  // Both vectors point to the same objects layers[] sees.
  std::vector<std::shared_ptr<BaseLayer>> enc_stack;
  std::vector<std::shared_ptr<BaseLayer>> dec_stack;
  std::shared_ptr<MatrixXf> encoded; //< Pointer to encoded output


public:
  AutoEncoder(SampleMatrix& training_set,
              SampleMatrix& test_set,
              int batch_size):MLP(training_set,test_set,batch_size){};

  // Add a new layer stack and lock the previous ones
  void addStackLayer(LayerConfig config);

  // For only encoding matrices
  MatrixXf encode(MatrixXf& set);
   
  


};


#endif

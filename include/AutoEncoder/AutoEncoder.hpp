#ifndef AUTOENCODER_HPP
#define AUTOENCODER_HPP

#include "MLP/MLP.hpp"

using InterfacePtr=std::shared_ptr<LayerInterface>;
using LayerPtr=std::shared_ptr<BaseLayer>;

/**
 * @brief Autoencoder class 
  */
class AutoEncoder:public MLP{
protected:
  // Stack structure for layers: enc_stack[0]==in / dec_stack[0]==out
  // Both vectors point to the same objects layers[] sees.
  std::vector<LayerPtr> enc_stack;
  std::vector<LayerPtr> dec_stack;
  // These don't contain the encoded interface
  std::vector<InterfacePtr> enc_interfaces;
  std::vector<InterfacePtr> dec_interfaces;
  InterfacePtr encoded_product=nullptr; //< Pointer to encoded output

  // Move the 2 stacks to the layers
  void convertToMLP();

public:
  AutoEncoder(SampleMatrix& training_set,
              SampleMatrix& test_set,
              int batch_size):MLP(training_set,test_set,batch_size){};

  // Add a new layer stack and lock the previous ones
  void addInterfaceStack(InterfacePtr new_encoded_interface);
  void addLayerStack(LayerProperties properties);

  // For only encoding matrices
  MatrixXf encode(MatrixXf& set);
   
  


};

#endif

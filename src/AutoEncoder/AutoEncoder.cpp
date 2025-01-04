#include "AutoEncoder/AutoEncoder.hpp"
#include <iostream>


/**
void AutoEncoder::addStackLayer(LayerConfig config){
  // If the stack isn't the 1st, input param is overriden
  if(enc_stack.size()>0){
    InterfaceParams outIF=enc_stack.back()->outputParamRef();
    config.input_size=outIF.
  }
  
}


MatrixXf AutoEncoder::encode(MatrixXf& set){
  const int enc_size=enc_stack.size();
  const PassContext context{
    set,VectorXi::Zero(0)
  };
  for(int i=0;i<enc_size;i++){
    enc_stack[i]->forward(context);
  }
  return enc_stack[enc_size-1]->outputRef();
}
*/

#include <iostream>
#include "CommonLib/basicFuncs.hpp"
#include "CommonLib/basicStructs.hpp"
#include "CommonLib/eventTimer.hpp"
#include "CommonLib/cifarHandlers.hpp"
#include "MLP/ActivationFunctions.hpp"
#include "MLP/MLP.hpp"

#define INPUT_DIM 32
#define INPUT_CHANNEL 3

namespace E=Eigen;
using InterfacePtr=std::shared_ptr<LayerInterface>;

int main(){
  EventTimer et;
  std::string dataset_path="../data/cifar-10-batches-bin";
  int training_size=5000;
  int test_size=1000;
  int batch_size=50;

  et.start("Load dataset");
  Cifar10Handler c10=Cifar10Handler(dataset_path);
  SampleMatrix training_set=c10.getTrainingMatrix(training_size);
  SampleMatrix test_set=c10.getTestMatrix(test_size);
  /*
  normalizeDataset(training_set.vectors,
                   test_set.vectors);
  */
  normalizeImageDataset(training_set.vectors, test_set.vectors, 3);
  et.stop();

  

  

  std::cout<<"Cool"<<std::endl;

  MLP mlp=MLP(training_set,test_set,batch_size);
  std::cout<<"Cool"<<std::endl;

  LayerProperties properties;
  properties.opt_config.type=Adam;
  properties.opt_config.adam.batch_size=batch_size;
  properties.opt_config.adam.rate=1e-3;
  properties.opt_config.adam.beta_1=0.9;
  properties.opt_config.adam.beta_2=0.999;
  properties.layer_type=FeedForward;

  std::vector<int> layer_sizes={1024,512,124};
  /*
  config.opt_config.type=SGD;
  config.opt_config.sgd.rate=1e-3;
  */
  InterfacePtr input=std::make_shared<LayerInterface>();
  input->width=input->height=32;
  input->channels=3;
  mlp.addInterface(input);

  std::cout<<"Cool"<<std::endl;

  for(auto l_size:layer_sizes){
    InterfacePtr interface=std::make_shared<LayerInterface>();
    interface->width=interface->channels=1;
    interface->height=l_size;
    interface->f=reLU;
    interface->f_dot=reLUder;
    mlp.addInterface(interface);
    std::cout<<"Inter"<<std::endl;
    mlp.addLayer(properties);
    std::cout<<"Layer"<<std::endl;
  }
  std::cout<<"Cool"<<std::endl;
  InterfacePtr interface=std::make_shared<LayerInterface>();
  interface->width=interface->channels=1;
  interface->height=10;
  interface->f=reLU;
  interface->f_dot=reLUder;
  properties.layer_type=SoftMax;
  mlp.addInterface(interface);
  mlp.addLayer(properties);

  float loss;

  et.start("Run epochs");
  std::cout<<"Epochs: "<<std::endl;
  for(int i=0;i<10;i++){
    loss=mlp.runEpoch();
    std::cout<<"Loss: "<<loss<<std::endl;
  }
  et.stop();

  et.displayIntervals();

}

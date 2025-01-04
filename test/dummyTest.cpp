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

int main(){
  EventTimer et;
  std::string dataset_path="../data/cifar-10-batches-bin";
  int training_size=2000;
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

  LayerConfig config;
  std::vector<int> layer_sizes={1024,512,124};
  config.layer_type=FeedForward;
  config.input_interface=std::make_shared<LayerInterface>();
  config.input_interface->height=config.input_interface->width=32;
  config.input_interface->channels=3;
  config.input_interface->size=32*32*3;
  config.f=reLU;
  config.f_dot=reLUder;
  config.opt_config.type=Adam;
  config.opt_config.adam.batch_size=batch_size;
  config.opt_config.adam.rate=1e-3;
  config.opt_config.adam.beta_1=0.9;
  config.opt_config.adam.beta_2=0.999;
  /*
  config.opt_config.type=SGD;
  config.opt_config.sgd.rate=1e-3;
  */



  for(auto l_size:layer_sizes){
    config.ff_config.output_sz=l_size;
    mlp.addLayer(config);
  }
  std::cout<<"Cool"<<std::endl;
  config.ff_config.output_sz=10;
  config.layer_type=SoftMax;
  mlp.addLayer(config);

  float loss;

  std::cout<<"Epochs: "<<std::endl;
  for(int i=0;i<10;i++){
    loss=mlp.runEpoch();
    std::cout<<"Loss: "<<loss<<std::endl;
  }


}

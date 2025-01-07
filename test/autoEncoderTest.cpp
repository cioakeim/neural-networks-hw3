#include <iostream>
#include "CommonLib/basicFuncs.hpp"
#include "CommonLib/basicStructs.hpp"
#include "CommonLib/eventTimer.hpp"
#include "CommonLib/cifarHandlers.hpp"
#include "MLP/ActivationFunctions.hpp"
#include "MLP/MLP.hpp"
#include "AutoEncoder/AutoEncoder.hpp"

#define INPUT_DIM 32
#define INPUT_CHANNEL 3

namespace E=Eigen;
using InterfacePtr=std::shared_ptr<LayerInterface>;


int main(){
  EventTimer et;
  std::string dataset_path="../data/cifar-10-batches-bin";
  std::string log_path_base="../data/AutoEncoder/master_run2";
  int training_size=50000;
  int test_size=10000;
  int batch_size=50;
  
  std::vector<int> epochs_list={30};
  std::vector<float> rates={5e-4};

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

  
  for(auto& epochs: epochs_list){
    for(auto& rate: rates){
      std::string log_path=
        log_path_base+"/"+std::to_string(epochs)+"_"+
        std::to_string(rate);
    
      AutoEncoder aenc=AutoEncoder(training_set,test_set,batch_size);
      std::cout<<"Cool"<<std::endl;

      LayerProperties properties;
      properties.opt_config.type=Adam;
      properties.opt_config.adam.batch_size=batch_size;
      properties.opt_config.adam.rate=5e-4;
      properties.opt_config.adam.beta_1=0.9;
      properties.opt_config.adam.beta_2=0.999;
      /*
      properties.opt_config.type=SGD;
      properties.opt_config.sgd.rate=1e-3;
      */
      properties.layer_type=MSE;

      InterfacePtr input=std::make_shared<LayerInterface>();
      input->width=input->height=32;
      input->channels=3;
      input->f=reLU;
      input->f_dot=reLUder;
      aenc.addInterfaceStack(input);
      std::cout<<"Cool"<<std::endl;

      std::vector<int> layer_sizes={512,124};

      ensure_a_path_exists(log_path);
      for(auto layer_size: layer_sizes){
        // Log the run with these layers
        std::ofstream log(log_path+"/run_"+std::to_string(layer_size)+".csv");
        if(!log.is_open()){
          std::cerr<<"CAN'T OPEN LOG: "<<log_path+"/run_"+std::to_string(layer_size)+".csv"<<std::endl;
          exit(1);
        }
        log<<"Epoch,J_train,J_test"<<"\n";
        InterfacePtr interface=std::make_shared<LayerInterface>(*input);
        interface->height=layer_size;
        interface->width=interface->channels=1;
        aenc.addInterfaceStack(interface);
        std::cout<<"Cool"<<std::endl;
        aenc.addLayerStack(properties);
        std::cout<<"Cool"<<std::endl;
        float J_train,J_test;
        float accuracy;
        et.start("Run epochs");
        std::cout<<"Epochs: "<<std::endl;
        for(int i=0;i<epochs;i++){
          J_train=aenc.runEpoch();
          std::cout<<"Loss: "<<J_train<<std::endl;
          aenc.testModel(test_set, J_test, accuracy);
          std::cout<<"Test loss: "<<J_test<<std::endl;
          log<<i<<","<<J_train<<","<<J_test<<"\n";
        }
        log.close();
        et.stop();
        properties.layer_type=FeedForward;
        //properties.opt_config.adam.rate=5e-5;
      } 

    }
  }

  et.displayIntervals();

}

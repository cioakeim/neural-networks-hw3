import numpy as np
import matplotlib.pyplot as plt
import glob
import re


def extractPCA(filename):
    fig, axs = plt.subplots(3, 1, figsize=(10, 8))
    data=np.loadtxt(filename,delimiter=",",skiprows=1)

    axs[0].plot(data[:,0],data[:,1])
    axs[1].plot(data[:,0],data[:,2])
    axs[2].plot(data[:,0],data[:,3])

    fig.suptitle("PCA Performance results")

    axs[0].set_yscale('log')
    axs[1].set_yscale('log')
    axs[2].set_yscale('log')

    axs[0].set_xlabel("# of components used")
    axs[1].set_xlabel("# of components used")
    axs[2].set_xlabel("# of components used")

    axs[0].set_ylabel("Info percentage kept")
    axs[1].set_ylabel("Training set MSE")
    axs[2].set_ylabel("Test set MSE")

    fig.tight_layout()
    plt.show()

    fig.savefig("pca_plots.eps")


def concatLogFileTo1csv(folder_name):
    csv_files=glob.glob(folder_name+"/run_*.csv")
    print(csv_files)
    possible_sizes=[1024,512,256,124]
    final_csv=np.ndarray([0,2])
    for size in possible_sizes:
        current_csv_name=folder_name+"/run_"+str(size)+".csv" 
        print(current_csv_name)
        if current_csv_name in csv_files:
            data=np.loadtxt(current_csv_name,delimiter=",",skiprows=0)
            if data.dtype!=np.float64:
                data=data[1:,:]
            data=data[:,1:3]
            final_csv=np.vstack((final_csv,data))
    fine_tune_name=folder_name+"/run_fine_tune.csv"
    if fine_tune_name in csv_files:
        data=np.loadtxt(fine_tune_name,delimiter=",",skiprows=1)
        data=data[:,1:3]
        final_csv=np.vstack((final_csv,data))
    rows = final_csv.shape[0]
    enumerator_column = np.arange(rows).reshape((rows,1))
    final_csv=np.hstack((enumerator_column,final_csv))
    return final_csv


def multipleRunsWithLabels(folder_list,label_list,title,
                           save_filename):
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))
    for folder,label in zip(folder_list,label_list):
        csv=concatLogFileTo1csv(folder)
        axs[0].plot(csv[:,0],csv[:,1],label=label)
        axs[1].plot(csv[:,0],csv[:,2],label=label)
    fig.suptitle(title)
    axs[0].set_xlabel("Number of epochs")
    axs[0].set_ylabel("Training set MSE")
    axs[1].set_xlabel("Number of epochs")
    axs[1].set_ylabel("Test set MSE")
    axs[0].legend()
    axs[1].legend()
    fig.tight_layout()
    plt.show()
    fig.savefig(save_filename+".eps")



def depthPlots(fileroot):
    depths=["shallow","deep"]
    folder_list=[]
    for depth in depths:
        folder_list.append(fileroot+"/"+depth+"/logs")
    multipleRunsWithLabels(folder_list,depths,
                           "Depth expertiment results",
                           "lock_depth")

def linearPlots(fileroot):
    layers=["1Layer","2Layer"]
    labels_list=["Single stack","2 Stacks"]
    folder_list=[]
    for layer in layers:
        folder_list.append(fileroot+"/"+layer+"/logs")
    multipleRunsWithLabels(folder_list,labels_list,
                           "Linear AutoEncoder results",
                           "nolock_linear")

def mlpPlot(fileroot):
    labels_list=["MLP"]
    folder_list=[fileroot+"/mlp_run"]
    multipleRunsWithLabels(folder_list,labels_list,
                           "Normal MLP results",
                           "nolock_mlp")

def rateSweepPlots(fileroot):
    subfolders_list=["try1","try2","try3","try4"]
    labels_list=["1e-4","2.5e-4","7.5e-4","1e-3"]
    folder_list=[]
    for sub in subfolders_list:
        folder_list.append(fileroot+"/"+sub+"/logs")
    multipleRunsWithLabels(folder_list,labels_list,
                           "Learning rate sweep results",
                           "lock_rate")


def lreluPlots(fileroot):
    layers=["2Layer","3Layer"]
    labels_list=["2 Stacks","3 Stacks"]
    folder_list=[]
    for layer in layers:
        folder_list.append(fileroot+"/"+layer+"/logs")
    multipleRunsWithLabels(folder_list,labels_list,
                           "Leaky ReLU results",
                           "lock_lrelu")

def batchNormPlots(fileroot):
    depth_list=["deep","shallow"]
    #batch_list=["batch_large","batch_small"]
    batch_list=["rate_large","rate_small"]
    batch_labels=[", Batch size: 200",", Batch size: 100"]
    file_list=[]
    label_list=[]
    for depth in depth_list:
        for batch,label in zip(batch_list,batch_labels):
            #file_list.append(fileroot+"/"+depth+"/"+batch+"/40epochs/logs")
            file_list.append(fileroot+"/"+batch+"/"+depth+"/try1/logs")
            label_list.append(depth+label)
    multipleRunsWithLabels(file_list,label_list,
                           "Batch normalization results",
                           "lock_batch_norm_enc_only")
    


def main():
    folder_name="../../../jobs/configs_batch_norm/rate_large/shallow/try1/logs"
    aenc_root="../../../AutoEncoder"
    aenc_root="../../../jobs/configs_locked"
    aenc_root="../../../jobs/configs_batch_norm_only_enc"
    depth_root=aenc_root+"/depth"
    linear_root=aenc_root+"/linear"
    rate_root=aenc_root+"/rate"
    lrelu_root=aenc_root+"/leaky"
    #depthPlots(depth_root)
    #linearPlots(linear_root)
    #mlpPlot(aenc_root)
    #rateSweepPlots(rate_root)
    #lreluPlots(lrelu_root)
    batchNormPlots(aenc_root)




if __name__=="__main__":
    main()

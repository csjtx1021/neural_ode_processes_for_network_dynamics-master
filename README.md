
## Running Environment and Dependency Packages

We have tested the code on the hardware environment with Intel(R) Xeon(R) Silver 4310 CPU@2.10GHz * 48 and NVIDIA GeForce RTX 3090 * 3.

The running environment and dependency packages are listed below.

    Python 3.10.8

        pickle==4.0
        
        numpy==1.23.5
        
        scipy==1.9.3
        
        networkx==2.8.4
        
        torch==1.13.1
        
        torch_scatter==2.1.0+pt113cu117
        
        sklearn==1.2.0
        
        pandas==1.5.2
        
        seaborn==0.12.2

	fastdtw==0.3.4

## Script Description for Running Code

### Training Phase: Train the NDP4ND for each network dynamics scenario

    bash runme_train.sh

### Test on all scenarios:

    bash runme_test.sh

### Test with various sparsities:

    bash runme_test_testSparsity.sh

### Test with various noises:

    bash runme_test_testNoise.sh

### get the infomation of testing results:

    python stat_info_of_tesing_results.py

***
NOTE:

* The generated training and testing sets can be found in ./data/DynamicsData/

* Trained models can be found in ./saved_models/

* Testing results can be found in ./results/
***




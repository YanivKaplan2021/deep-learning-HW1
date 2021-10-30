## deep-learning-HW1
In order to install all needed libraries:

> pip3 install -f requirements.txt

If torch libraries are not found on pip, try downloading them from Conda.

Without GPU: 
> conda install pytorch torchvision torchaudio cpuonly -c pytorch

With GPU: 
> conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch

run.py is the main code to run.

You can tweak the configuration file config.yaml:
> use_gpu: True / False

> learning_rate: 0.01 (double)

> num_epochs: 12 (integer)

> batch_size: 100 (integer)


> dropout:
>>      use: True
>>      dropout_ratio: 0.2 (double)


> use_batch_normalization: True/ False


> weight_decay:
>>      use: True/ False
>>      parameter: 0.01 (double)


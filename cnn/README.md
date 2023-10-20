# Coverage Testing
This is a Repository of neuron coverage testing for CNN, RNN, and DRL.
So Far, the accomplishment subjects includes:
 - CNN
## Project Architecture
    /attack_images: implementations of generating adversarial images. the adv images will be saved in /data/dataset_name/adv/model_name/attack_name/xxx.test.npy
     (If you want to generage new adv images, use command like this: python attack.py  --dataset cifar  --model vgg16 --attack fgsm --batch_size 128)
    /coverage: the result of the generated result.
    /data: the dataset being used. (origin dataset + adv images)
    /models: the models being used.
    /neuron_coverage_cnn: implementations of cnn neuron coverage.

## 1. For CNN
    /neuron_coverage/coverage_curve.py
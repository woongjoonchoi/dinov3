from torchvision.datasets import Caltech256 ,Caltech101 ,CIFAR100,CIFAR10,MNIST,ImageNet


# exit()


if __name__ =='__main__' :
    # print(os.getcwd())
    # imagenet_dataset = ImageNet(root='ImageNet',split='train')
    # /datasets/imagenet
    # imagenet_dataset = ImageNet(root='imagenet/train',split='train')
    imagenet_dataset = ImageNet(root='/datasets/imagenet',split='val')
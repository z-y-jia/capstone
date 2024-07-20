

# generative models 

# type


## VAE



## GANs

# motivation of GAN

## VAE(Variational Autoencoder)

VAE是一种深度学习模型，它可以学习一个给定的分布，通过比较输入和输出来实现。

VAE可以学习数据的潜在表示。

VAE的生成能力较差，因为VAE学习的是数据的平均表示。

VAE的输出通常是模糊的。
Variational Autoencoder (VAE)
• VAE learns a given distribution by comparing its input
to its output.
• VAE is good for learning hidden representations of data.
• VAE is pretty bad for generating new data, because
VAE learns an “averaged” representation of the data.
• VAE’s output is usually blurry


## Generative Adversarial Network (GAN)
• GAN is a deep learning model that consists of a generator and a discriminator.
• The generator is a network that can generate new samples, while the discriminator is a network that can determine whether an input is real or fake.
• The generator’s goal is to generate samples that are as close as possible to the real data, while the discriminator’s goal is to distinguish between real and generated samples.
• GAN’s training process involves a two-stage process: training the generator and training the discriminator.
• The generator’s loss function tries to minimize the discriminator’s ability to correctly classify real and generated samples, while the discriminator’s loss function tries to maximize the generator’s ability to fool the discriminator.
• GANs can generate high-quality images, because they can generate realistic images.
• GANs can generate different types of images, because they can generate images of different types.

## GAN vs VAE

VAE和GAN都可以学习数据分布，但是它们的目标不同。

VAE的目标是学习数据的潜在表示，所以它学习的分布是离散的。

GAN的目标是生成尽可能真实的样本，所以它学习的分布是连续的。

VAE的生成能力较差，因为VAE学习的是数据的平均表示，所以生成的样本可能不真实。

GAN的生成能力较好，因为GAN可以生成逼真的图像。

- Generative Adversarial Networks (GANs) are a type of deep learning model that can be used for generating new data samples. 
- GANs are based on the idea of a two-player game, where a generator network learns to create new data samples that are indistinguishable from the training data, while a discriminator network learns to distinguish between the real and generated data. 
- The generator network is trained to fool the discriminator network, which means that it produces data that is indistinguishable from the training data. 
- The discriminator network is trained to distinguish between the real and generated data, so that it can correctly classify the data as real or generated. 
- GANs can be used for a variety of tasks, such as image synthesis, text-to-image synthesis, and video synthesis.


## stable diffusion models

- Stable diffusion models are a type of generative model that can be used for generating new data samples. 
- Stable diffusion models are based on the idea of diffusion processes, which are processes that describe how information is spread through a system. 
- Stable diffusion models use a Markov chain to simulate the diffusion of information through a system. 
- The Markov chain is a stochastic process that describes the probability of transitioning from one state to another. 
- Stable diffusion models can be used for a variety of tasks, such as image synthesis, text-to-image synthesis, and video synthesis.
# Example of straight-through gradient estimator of stochastic neural networks
Idea is based on stochastic neural networks with the straight-through gradient estimator

- Bengio, Yoshua, Nicholas Léonard, and Aaron Courville. "Estimating or propagating gradients through stochastic neurons for conditional computation." arXiv preprint arXiv:1308.3432 (2013).
- Hafner, Danijar, et al. "Mastering atari with discrete world models." arXiv preprint arXiv:2010.02193 (2020).

In pytorch, straight-through gradient is obtained using this equation in the forward inference:

$\hat h = h.detach() + p - p.detach()$,

where $h$ is the sample from the probability $p$.

## Task
digit images "3" or "4" ( $y$) are presented with a conditioning input  $x$ (one_hot. [1, 0] or [0, 1]) . 
The probability of target images are sampled by $y \sim p(y|x)$ .

In addition to the ordinary straight-through estimator, I added an entropy loss of the binary neuron in the training.

$L = L_{\rm STE} - H[p_b]$

### [3]:[4] = (1:9 @ x=0), (9:1 @ x=1)
![1-9](https://user-images.githubusercontent.com/1684732/188325134-cd613a60-ecf2-4308-beee-e50af9ad33b1.png)

### [3]:[4] = (2:8 @ x=0), (8:2 @ x=1)
![2-8](https://user-images.githubusercontent.com/1684732/188325271-366139c5-5f26-4bb0-acc9-00237548c47a.png)

### [3]:[4] = (0:10 @ x=0), (10:0 @ x=1)
![0-10](https://user-images.githubusercontent.com/1684732/188325284-c331b977-921d-40b8-853a-267318491ecd.png)

## Some unsatisfactory results

### [3]:[4] = (4:6 @ x=0), (6:4 @ x=1)
![4-6](https://user-images.githubusercontent.com/1684732/188325313-6013e451-ab0b-4207-b5f1-2b85c7e5f486.png)

### [3]:[4] = (10:0 @ x=0), (4:6 @ x=1)
![スクリーンショット 2022-09-05 2 17 41](https://user-images.githubusercontent.com/1684732/188325518-8d556934-5b1c-478e-afef-9a8fb176f2bd.png)

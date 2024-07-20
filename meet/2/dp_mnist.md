
# map f:X->Y
N: number of particles(samples),
M is the number of pixels per dimension
$
X=\{(x_1,x_2,\eta)\}_N , (x_1,x_2)\sim Unif(F,F),\\
F=\{0,1, \ldots ,M-1\},\\
\\
\to\\ 
Y=f(X)=\{(y_1,y_2)\}_N, f=\{g_\eta\}\\
(y_1,y_2)=net(x_1,x_2,\eta) \sim g_\eta(F,F)
$

# mnist dataset:
image: 28x28 pixels, grayscale
label: 10 classes

$
X=\{(x_1,x_2)\}_N, (x_1,x_2)\sim Unif([0,27],[0,27]),\\
\eta=0,1,2,3,4,5,6,7,8,9\\
\to\\
Y=f(X)=\{(y_1,y_2)\}_N, f=\{g_\eta\}\\
(y_1,y_2)=net(x_1,x_2,\eta) \sim g_\eta([0,27],[0,27])\\
$


$
eg. \eta=0, X=\{(x_1,x_2)^0\}={(1,2),(3,4)}, net(X)=\{(5.1,6.7),(7.8,9.4)\}=\{(5,6),(7,9)\}\\
$













layers = [m_in, 20, 20, 20, 20, 20, 20,
          20, 20, 20, 20, 20, 20, m_out]
m_in是3维的，m_outs是2维，如何从m_in中提出1维，放到layer1_2,layer2_2,layer3_2这三层中，其中前三层每一层的输入都是3维，而从4层之后都变成2维。layer1_2,layer2_2,layer3_2和layer1_1,layer2_1,layer3_1之间有交叉的链接，也就是layer2_2输入是layer1_1的输出，而layer2_2输出





1. cross W Dist: lang->vision
motivation: 图片中的文字乱写

2 step ARBCD less particles: 100生成模糊图像，再来100模糊图像到精确图（补全断点）

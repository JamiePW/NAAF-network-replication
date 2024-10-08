该网络是一个具有3个一维卷积层的ResNet，在Speck各类攻击中使用的均是同一结构，但有3项参数略有不同：

`length`是输入层的长度，从使用情况来看应该为4的倍数，因为输入后会将一维数据重整为`(4, length//4)`的二维形式

`filters`代表三个卷积层中滤波器/卷积核的数量，在针对Speck的攻击中，三个卷积层的此项参数都是相同的

`kernel_size`设定了第二个和第三个卷积层的滤波器/卷积核大小，而第一层的卷积核大小始终为1

下表展示了NAAF开源项目中针对Speck的不同攻击中该网络的参数设定供参考

| 攻击场景  | `length` | `filters` | `kernel_size` |
| --------- | -------- | --------- | ------------- |
| 64_6_42   | 40       | 32        | 3             |
| 64_6_47   | 48       | 32        | 3             |
| 64_6_33   | 40       | 32        | 3             |
| 96_7_53   | 48       | 48        | 5             |
| 96_7_65   | 48       | 48        | 5             |
| 96_7_77   | 48       | 48        | 5             |
| 96_7_89   | 48       | 48        | 5             |
| 128_9_64  | 44       | 32        | 5             |
| 128_9_76  | 44       | 32        | 5             |
| 128_9_90  | 48       | 32        | 5             |
| 128_9_105 | 48       | 32        | 5             |
| 128_9_117 | 20       | 32        | 5             |


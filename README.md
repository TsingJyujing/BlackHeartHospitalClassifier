# 东云研究所——如何利用机器学习鉴别莆田系医院名称

## 简介

自从百度魏则西事件以后，我持续在关注我身边的医院，我发现莆田系的医院大多是有命名的特征的（承包科室的那种真的不能识别了）。

我随便给出几个“很莆田”的名字：

“XX玛利亚女子医院”
“XX女子医院”
“XX男科医院”
“XX肛肠医院”
“XX阳光医院”

前几天我和同事在车上说起这个事情，同事纷纷给出了自己鉴别一些莆田系医院的经验，例如莆田系特别喜欢找“武警医院”下手。
我们可以看出，莆田系医院是有一定特征的，光靠医院列表我们无法鉴别如雨后春笋新开的莆田系医院，一个人的经验是有限的，我们如何鉴别这些黑心医院呢？

没错！这次的主题还是 Machine Learning 大法好。

但是我并不是一开始就决定要用 Machine Learning 的，最早打算使用词表来做，但是考虑到中国地方太大，医院有一万多个，整理出来的莆田系医院也有一千多个，辛辛苦苦整理关键字绝对不是我的Style。

## RNN

对于短句分类，直觉里首选的肯定是RNN（LSTM也算）啦～毕竟算是一个序列的问题。我们通过构建OneHot编码构建向量，随后送入RNN进行训练，每一个字即运行一个Step，直到最后输出结果再进行反馈。属于多输入单输出的问题。

## RNN的问题

刚开始的时候，打算使用 Pytorch（其实我还写过一个[Java版的LSTM](https://github.com/TsingJyujing/java_rnn)）。
结果偷懒了就直接去抄官方的代码，正好Pytorch的代码里有这个么一个教程：
[CLASSIFYING NAMES WITH A CHARACTER-LEVEL RNN](https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html)

但是这个教程和真实情况略有写不同，需要针对本问题做一些优化：

1. 中文的Charset和英文不一样，中文常用字有6k个，如果组词的话就更多了。之前考虑用字嵌入（其实就是用词嵌的方法处理单字中文语料）。
    - 但是本案例样本量太小，使用其他语料训练不一定能学出这些字在这个语境下的关系
2. 例子里没有划分训练测试集合，没有做Early Stopping，容易过拟合
    - 但是由于英文字母数量少，所以原来的教程不容易过拟合

上了个厕所洗了个澡（这就思考整整两遍人生了），认为RNN有一个更严重的问题：可解释性不好，难以将学习到的东西重新整理为人类能用的经验，所以大家每次要鉴别医院都要打开我的页面。

## Sparse Logistic Regression

我们考虑将分词后所有的词语收集起来，然后使用OneHot编码，这个时候我们得到了一个稀疏向量，多个样本构成稀疏矩阵，使用林老师的liblinear就可以学习啦～（当然最后图省事用的sklearn调liblinear）。

其实中途还尝试过使用SVD先对数据进行一个降维操作，但是实际效果很不好。和之前不用词嵌的理由一样：**在有监督的数据前面，做一层无监督的降维，绝壁是会降低精度的行为，Absofuckinglutely。**

### 预处理
不需要预处理，因为数据本身已经被Normalize了。这里也不需要过采样，因为我们在训练的时候设置了权重。

### 结果评价

结果评价的一般方法我写在了这里：[机器学习中结果评估与模型解释](https://zhuanlan.zhihu.com/p/40718648)（其实这篇才是我在工作中呕心沥血总结出来的干货，不过居然没人看，难道大家都去看xhamster了吗？伤心……）

### 可解释性
想要可解释，我们就必需在人脑内能建立一个简单的模型，幸运的是，(Sparse) Logisic Regression 很好理解，如果不能理解的，可以去看我以前的一些文章：

- [Logistic回归的CPU并行化及其 Batch Gradient Ascent 的实现 (这篇有原理)](https://zhuanlan.zhihu.com/p/20511129)
- [如何使用Logistic回归做非线性分类](https://zhuanlan.zhihu.com/p/20545718)
- [瞎玩儿系列 使用SQL实现Logistic回归](https://zhuanlan.zhihu.com/p/26739934)
- [补上之前被和谐的坑](https://zhuanlan.zhihu.com/p/25764539)
    - 这篇是实战篇
    - 实际上是：[很污的机器学习：从xhamster网站找到喜欢的片子](http://www.cnblogs.com/TsingJyujing/p/6549780.html)

我们只需要找出参数最大或者最小的几项就可以了鸭～
让我们看看你们都应该注意哪些关键词呢？

存在这些关键词的，你们避而远之：
```python
['慧慈', '凉山', '玛莉亚', '爱德华', '济南', '阳光', '国丹', '九州', '天骄', '莆田', '江南', '青羊区', '南亚', '妇科', '中骏']
```
当然，地名是不需要你们关心的，那么还剩下：
```python
['慧慈', '玛莉亚', '爱德华', '阳光', '国丹', '九州', '天骄', '莆田', '江南', '南亚', '妇科', '中骏']
```
是不是满满的莆田感？

同样的操作之后我们也可以得到让人放心的关键字：

```python
['264', '中国人民武装警察部队', '钢铁', '中国人民解放军', '分院', '协合', '护理', '铁路分局', '医学院', '职工', '乳房', '地段', '妇幼保健']
```

### 改进
对句子的分析应该做更深入的定制化，医院的名称都有一定的格式，例如`XX省XX市（当然地名一般是不关注的）XX大学附属医院XX分院`之类的，通过对句子更加细致的分析，我们可以得到一个K维矢量（这个K很小），决定K大小是对句子的细分程度，例如是否是大学附属？哪个大学？是否是分院？名称是否有某个科室的名字（如肿瘤，骨科）？科室名字是什么？

因为效果还行，而且仅仅是通过名字判别也不需要特别准确，这个觉得有疑惑的时候，上一下百度，有推广的医院基本都有点问题。再用查询工商信息的软件查一下就可以判断的七七八八了。

因为有的莆田系医院就是正常医院的科室被承包以后形成的，莆田系医院是一个社会问题，仅仅凭借名字是无法百分百判断的。
如果名称的秘密被大家接受，莆田系医院只会开的更加隐蔽，更加险恶。

## 页面化

最后我做了个小工具，挂在了东云研究所的子域名下，页面做的很丑，但是可以玩一下，前后端分离，大家也可以调用我的接口，在服务器给你们玩挂之前我不会限流。

访问：[http://hospital-classifier.shinonomelab.com](http://hospital-classifier.shinonomelab.com) 来利用小工具，或者调用API接口。

接口地址：`http://hospital-classifier.shinonomelab.com/api/classifier/?name=医院名称`

返回结果说明：

```js
{
  "status": "success",// success: 正常识别了 fail：识别过程出现错误
  "result": true,// true：正常医院 false：不正常医院
  "input": "江阴市人民医院" //输入内容的回显
}
```

## 讨论与改进

其实我读了好多论文，但是很多方法并不适用于本问题，有的是因为数据量不够大，有的是因为问题种类不对。

总结下来，有可能的改进我写在下面：

- 尚未对“这个名称不是医院”这一情况进行学习
- 使用医疗新闻进行词嵌以后取词向量进行学习
- 针对医院名称做专门的句法分析
- 使用接口获取百度推广相关信息
- 使用接口获取股东信息和诉讼信息

如果你有什么想法，请在评论区提出，也欢迎私下联系我讨论：[TsingJyujing/Resume#basic-info](https://github.com/TsingJyujing/Resume#basic-info)。

## 附录

### 数据来源
- [莆田系医院名单](https://github.com/shenwei356/BlackheartedHospital)
- [全国医院列表](http://www.a-hospital.com/w/%E5%85%A8%E5%9B%BD%E5%8C%BB%E9%99%A2%E5%88%97%E8%A1%A8)

### 部分参考文献、资料

（有些看了，但是没留链接……就不一一列出了）

- [CLASSIFYING NAMES WITH A CHARACTER-LEVEL RNN](https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html)
- [Improved Semantic Representations From Tree-Structured Long Short-Term Memory Networks](https://arxiv.org/abs/1503.00075)
- [word2vec Explained: deriving Mikolov et al.'s negative-sampling word-embedding method](https://arxiv.org/abs/1402.3722)
- [The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)
- [Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)

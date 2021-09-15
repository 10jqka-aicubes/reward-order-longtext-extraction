# 		【9-10双月赛】超长答案的阅读理解			

​	机器阅读理解是根据问题从文本中自动地抽取答案的任务。目前主流的NLP模型如BERT只适合抽取答案小于固定长度（512）的数据。然而，在一些真实业务中，答案往往是该长度的几倍。因此，需要一种适合长文本抽取的模型来解决此类业务，该模型需要克服以下难题：1.随着句子长度增加，模型的计算复杂度成倍增加；2. 文本过长导致信息遗忘。

​	本次任务我们提供诉讼公告的文本数据，需要参赛人员完成机器学习/深度学习系统，将公告中的违规案例抽取出来，一篇公告中的违规案例可能会包含多个答案。

​	本代码是该赛题的一个基础demo，仅供参考学习。

​	比赛地址：http://contest.aicubes.cn/	

​	时间：2021-09 ~ 2021-10



## 如何运行Demo

- clone代码


- 准备预训练模型

  - 下载模型 [chinese-roberta](https://huggingface.co/hfl/chinese-roberta-wwm-ext/resolve/main/pytorch_model.bin)
  - 将文件放在`./model/roberta_zh/0_Transformer/`

- 准备环境

  - cuda10.0以上
  - python3.7以上
  - 安装python依赖

  ```
  python -m pip install -r requirements.txt
  ```

- 准备数据，从[官网](http://contest.aicubes.cn/#/detail?topicId=24)下载数据

  - 将训练数据随机按9:1切分为`train.json`和`valid.json`，放在训练数据目录中
  - 预测数据`content_test.json`放在预测目录下

- 调整参数配置，参考[模板项目](https://github.com/10jqka-aicubes/project-demo)的说明

  - `reward_order_longtext_extraction/setting.conf`
  - 其他注意下`run.sh`里使用的参数

- 运行

  - 训练

  ```
  bash reward_order_longtext_extraction/train/run.sh
  ```

  - 预测

  ```
  bash reward_order_longtext_extraction/predict/run.sh
  ```

  - 计算结果指标

  ```
  bash reward_order_longtext_extraction/metrics/run.sh
  ```



## 反作弊声明

1）参与者不允许在比赛中抄袭他人作品、使用多个小号，经发现将取消成绩；

2）参与者禁止在指定考核技术能力的范围外利用规则漏洞或技术漏洞等途径提高成绩排名，经发现将取消成绩；

3）在A榜中，若主办方认为排行榜成绩异常，需要参赛队伍配合给出可复现的代码。



## 赛事交流

加入![同花顺比赛小助手](http://speech.10jqka.com.cn/arthmetic_operation/245984a4c8b34111a79a5151d5cd6024/客服微信.JPEG)
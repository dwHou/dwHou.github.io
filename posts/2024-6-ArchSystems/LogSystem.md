## 介绍

Gl\*\*alTra\*\*\*g（Zebra）是我司内部的一个日志查询平台。

## 背景

日志查询是为Troubleshooting服务的。

**1. Troubleshooting痛点**

- 由于安全问题，国际员工不允许查询产线Log
- ES（Amazon Elasticsearch）集群很多，要判断数据落在哪个ES集群，然后打开相应的Kibana（Amazon的日志查询系统）地址进行查询
- 不支持某些Log导出
-  微服务架构下，交易跨多个系统，Troubleshooting较困难

**2. 基础架构**

一般来说是 ：

Service → Kafka Topic → Kafka Log Intake → ES集群 → 日志查询平台

>  [!NOTE]
>
> - Kafka是LinkedIn开源的日志收集系统，Topic 是 Kafka 中最基本的数据单元。
> - 我司使用自研的创建topic和采集日志的Agent。
> - LogSystem一般来说包括Log Intake、Log Storage、Log Analysis、Log Management等组件，其中Log Intake是前端收集日志的组件，而其他组件在后端。这些组件有些可以自研，有些起初可以先用开源的项目。
> - 被分析后的日志（parsed log）推送到ES。
>
> - 其中 → 表示被消费。

**3. 如何解决痛点**

- 指定安全规则：Log分级和Role based的权限控制。
- 统一查询入口：只提供一个查询站点，并行请求相关的ES集群进行Search。
- 支持Log导出：生成的日志文件上传到相应地区的S3（Amazon Simple Storage Service，是亚马逊提供的一种对象存储服务），以便随时下载。
- 引入全链路追踪Trace：需要埋点（接入一下Trace Agent即可），产生不同于业务Log的另一种链路追踪的Log，可以令Troubleshooting更高效。

>  [!NOTE]
>
> 我的评价：我司的日志查询平台最常用的Discover（日志查询）功能，其实和Kibana的差不多。而Kibana设计上是指定地址来查询，我们是依次选择地区和服务，这是抽象和定制化的区别。


# 创建镜像

当我们从 docker 镜像仓库中下载的镜像不能满足我们的需求时，我们可以通过以下两种方式对镜像进行更改。

- 1、从已经创建的容器中更新镜像，并且提交这个镜像
- 2、使用 Dockerfile 指令来创建一个新的镜像

## 壹：更新镜像

更新镜像之前，我们需要使用镜像来创建一个容器。

```shell
devonn@devonn:~$ docker run -t -i ubuntu:15.10 /bin/bash
root@e218edb10161:/# 
```

在运行的容器内使用 **apt-get update** 命令进行更新。

在完成操作之后，输入 exit 命令来退出这个容器。

此时 ID 为 e218edb10161 的容器，是按我们的需求更改的容器。我们可以通过命令 docker commit 来提交容器副本。

```shell
devonn@devonn:~$ docker commit -m="has update" -a="devonn" e218edb10161 devonn/ubuntu:v2
sha256:70bf1840fd7c0d2d8ef0a42a817eb29f854c1af8f7c59fc03ac7bdee9545aff8
```

各个参数说明：

- **-m:** 提交的描述信息
- **-a:** 指定镜像作者
- **e218edb10161：**容器 ID
- **devonn/ubuntu:v2:** 指定要创建的目标镜像名

## 贰：构建镜像

我们使用命令 **docker build** ， 从零开始来创建一个新的镜像。为此，我们需要创建一个 Dockerfile 文件，其中包含一组指令来告诉 Docker 如何构建我们的镜像。

```shell
# 这是一个示例的 Dockerfile

# 基于 Ubuntu 作为基础镜像
FROM ubuntu:latest

#  LABEL 指令来添加元数据信息，包括维护者的信息。
LABEL maintainer="Devonn <devonn@gmail.com>"

# 安装必要的软件包
RUN apt-get update && apt-get install -y \
    package1 \
    package2

# 复制代码到镜像中
COPY code /app

# 设置工作目录
WORKDIR /app

# 运行应用程序
CMD [ "python", "app.py" ]
```

每一个指令都会在镜像上创建一个新的层，ChatGPT告诉我们以下指令常用：

1. `FROM`：指定基础镜像。用于指定构建镜像所使用的基础操作系统或其他镜像。
2. `RUN`：执行命令。用于在镜像构建过程中执行命令，例如安装软件包、运行脚本等。
3. `COPY`：复制文件或目录。用于将文件或目录从构建上下文复制到镜像中的指定位置。
4. `ADD`：复制文件或目录，并支持 URL 和解压缩。类似于 `COPY`，但还支持从 URL 复制文件，并在需要时自动解压缩压缩文件。
5. `WORKDIR`：设置工作目录。用于设置后续指令的工作目录，例如执行命令、复制文件等。
6. `ENV`：设置环境变量。用于在镜像中设置环境变量，可以在容器运行时被使用。
7. `EXPOSE`：声明暴露的端口。用于声明容器运行时将监听的端口，并提供给其他容器或主机访问。
8. `CMD`：指定容器默认命令。用于定义容器启动时默认要执行的命令，可以被 `docker run` 命令的参数覆盖。
9. `ENTRYPOINT`：设置容器入口点。类似于 `CMD`，但它不会被命令行参数覆盖，而是作为容器的主要命令执行。
10. `VOLUME`：声明挂载点。用于声明容器中的目录，可以被其他容器或主机挂载为卷。

然后，我们使用 Dockerfile 文件，通过 docker build 命令来构建一个镜像。

```shell
docker build [OPTIONS] PATH
```

其中，`OPTIONS`表示可选参数，`PATH`表示包含Dockerfile的目录路径。

常用的`docker build`选项包括：

- `-t, --tag`：为镜像指定标签。可以使用`<仓库名>:<标签>`的格式，例如`myimage:1.0`。
- `-f, --file`：指定要使用的Dockerfile路径。默认情况下，`docker build`命令会在当前目录查找名为`Dockerfile`的文件，但你可以使用该选项来指定其他路径。
- `-rm`：在构建过程完成后自动删除中间容器。这可以帮助减少磁盘空间的使用。（在Docker镜像的构建过程中，每一条指令都会生成一个临时容器来执行该指令，并且在执行完成后保留这个中间容器。<font color="brown">使用`-rm`选项是一种良好的实践</font>）
- `--no-cache`：禁用构建过程中的缓存。默认情况下，Docker会尽可能使用缓存的中间镜像层来加速构建过程。使用该选项可以确保每次都重新构建。

就此镜像创建成功~

**设置镜像标签**

`docker tag` 命令用于给本地的镜像打标签，为其创建一个别名。这在将镜像推送到远程仓库或者在本地使用不同的标签来管理镜像时非常有用。语法如下：

```shell
docker tag SOURCE_IMAGE[:TAG] TARGET_IMAGE[:TAG]
```

其中，`SOURCE_IMAGE` 是要标记的源镜像，可以是镜像的名称或镜像ID。`:TAG` 是可选的标签，用于指定特定的版本或标识符。`TARGET_IMAGE` 是目标镜像的名称和可选的标签，用于创建新的标签。

## 叁：查看镜像

`docker history IMAGE_NAME:TAG` 可以查看docker镜像的构建历史；

`docker inspect IMAGE_NAME:TAG` 可以查看docker对象的元数据信息。

>尤其是当我们从一些基础镜像构建时，除了在dockerhub网站上查看，也可以用这些命令查看镜像，更为方便。


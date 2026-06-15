# PyTorch Image Generator

这是一个升级后的 `PyTorch` 图片生成项目，支持卷积版 `DCGAN` 和更重的 `WGAN-GP`，训练 `64x64` 图片，不再只是线性层生成数字。

它现在支持：

- `CPU` 训练
- `MPS` 训练（Apple Silicon 可选）
- `CIFAR10` 彩色图片生成
- `FashionMNIST` 灰度服饰生成
- `ImageFolder` 自定义图片数据集训练
- 自动保存采样图和检查点
- `heavy-cpu` / `overnight-cpu` 重负载训练档位

## 1. 这次升级了什么

旧版本：

- 只生成 `MNIST` 手写数字
- 只有全连接层
- 更像教学 Demo

新版本：

- 默认训练 `CIFAR10`
- 使用卷积生成器和卷积判别器
- 支持 `WGAN-GP` 重训练模式
- 输出 `64x64` 图片
- 可以切换到你自己的图片文件夹
- 可以通过线程和 `DataLoader workers` 把 `CPU` 压得更满

## 2. 适合你的机器吗

你的机器是 `MacBook Pro M1 Pro`，如果你坚持用 `CPU`：

- 这个项目可以明显比之前重很多
- 用 `CIFAR10 + 120 epochs` 跑起来会持续吃资源
- 如果改成你自己的大一点的数据集，完全可以跑很久

但要说清楚一点：

- “满负载跑 8 小时”不取决于框架本身
- 取决于数据集大小、训练轮数、模型宽度、batch size、线程数

所以我已经把这些参数都开放出来了。

## 3. 环境要求

建议使用：

- Python `3.10` / `3.11` / `3.12`

当前目录里的 `.venv` 是 `Python 3.13.5`。如果安装 `torch` 不顺，建议改用 `3.11` 虚拟环境。

## 4. 安装依赖

```bash
pip install -r requirements.txt
```

## 5. 开始训练

默认训练 `CIFAR10`：

```bash
python train.py --device cpu
```

如果你想让 `CPU` 更忙一些，可以这样：

```bash
python train.py --device cpu --dataset cifar10 --profile heavy-cpu
```

如果你愿意使用苹果芯片图形加速，可以改成：

```bash
python train.py --device mps
```

## 6. 训练你自己的图片

你的图片目录建议这样放：

```text
my_images/
└── train/
    ├── cats/
    │   ├── 001.jpg
    │   ├── 002.jpg
    └── dogs/
        ├── 003.jpg
        ├── 004.jpg
```

然后运行：

```bash
python train.py --device cpu --dataset imagefolder --data-root my_images/train --image-channels 3 --profile heavy-cpu
```

如果你的数据是灰度图：

```bash
python train.py --device cpu --dataset imagefolder --data-root my_images/train --image-channels 1
```

## 7. 单独生成图片

```bash
python generate.py --checkpoint outputs/checkpoints/generator_last.pt --num-images 16 --device cpu
```

## 8. 输出内容

- `outputs/samples/`：训练过程中生成的采样图
- `outputs/checkpoints/`：模型检查点
- `outputs/generated/`：单独采样生成的图

## 9. 怎样更容易跑到 8 小时

如果你的目标不是“先看效果”，而是“让机器持续高负载训练很久”，建议这样做：

- 优先用 `WGAN-GP`
- 用 `imagefolder` 训练你自己的大量图片
- 把 `epochs` 提到 `220`、`320` 甚至更高
- 把 `base-channels` 提到 `96` 或 `128`
- 增大 `critic-steps`
- 把 `batch-size` 调到你机器能承受的上限
- 把 `num-workers` 调到 `6-8`
- 把 `cpu-threads` 调到 `8-10`

最省事的两档：

```bash
python train.py --device cpu --dataset cifar10 --profile heavy-cpu
python train.py --device cpu --dataset imagefolder --data-root my_images/train --profile overnight-cpu
```

一个更重的手动例子：

```bash
python train.py --device cpu --dataset imagefolder --data-root my_images/train --gan-mode wgan-gp --epochs 360 --batch-size 144 --base-channels 128 --latent-dim 192 --critic-steps 6 --num-workers 8 --cpu-threads 10
```

## 10. 注意

`WGAN-GP` 会比 `DCGAN` 更重，也更适合你这种“故意让机器长时间高负载跑”的目标，但训练时间会明显变长。

## 11. 错误日志分类器

如果你现在要做“错误日志分类”，项目里已经加了一个可运行模板。

数据格式：

```csv
log_text,label
"java.sql.SQLTimeoutException: query timed out after 30000ms",database_timeout
"Permission denied for user admin on /api/login",auth_error
```

训练：

```bash
python train_log_classifier.py --data-path data/logs/train.csv --device cpu
```

预测：

```bash
python predict_log.py --text "socket hang up during request to payment gateway" --device cpu
```

相关文件：

- `train_log_classifier.py`：训练入口
- `predict_log.py`：单条日志预测
- `src/log_data.py`：数据读取、分词、词表
- `src/log_classifier.py`：`GRU` 文本分类模型

适合的标签例子：

- `database_error`
- `database_timeout`
- `network_error`
- `network_timeout`
- `auth_error`
- `disk_error`

真正上线时，最重要的不是模型多复杂，而是你的训练数据标签要稳定、一致、够多。

如果你后面想继续升级，下一步比较合理的是：

- `Conditional GAN`
- `WGAN-GP`
- `StyleGAN` 风格路线
- `Diffusion` 扩散模型路线

## 12. LAION 人像 URL 过滤工具

如果你要从 `LAION` 里筛“可下载的人像 URL”，项目里已经加了一个工具：

- [filter_laion_people.py](/Users/allenflux/PycharmProjects/ml/filter_laion_people.py)

这个工具会做几件事：

- 从 `LAION` 流式读取样本
- 用文本规则先筛“人、单人、全身、带脸”的候选 caption
- 真正请求 URL
- 过滤掉下载失败、不是图片、分辨率太小的样本
- 导出 `jsonl/csv`
- 可选把图片也下载到本地

运行示例：

```bash
python filter_laion_people.py --max-samples 5000 --max-keep 200 --download-images
```

结果默认保存在：

- `outputs/laion_people/filtered.jsonl`
- `outputs/laion_people/filtered.csv`
- `outputs/laion_people/images/`

要注意：

- 这个工具能保证 URL 可下载且内容是图片
- 但“单人、全身、带脸”目前是基于 caption 文本提示筛的，不是视觉检测 100% 保证
- 如果你要特别准，下一步应该再加人体检测和人脸检测

## 13. LAION 人像 API 服务

如果你要“调用一个 API 就拿到合格图像下载链接”，现在已经加了服务版：

- `laion_people_api.py`
- `src/visual_person_filter.py`
- `run_laion_people_api.sh`

它的流程是：

- 后台持续随机扫描 `LAION`
- 先用文本筛候选
- 下载图片并确认 URL 可用
- 用 OpenCV 做人脸检测和人体检测
- 通过后保存到本地缓存
- API 随机返回一个本地可下载链接

安装依赖：

```bash
pip install -r requirements.txt
```

启动服务：

```bash
./run_laion_people_api.sh
```

或者手动启动：

```bash
python laion_people_api.py --host 127.0.0.1 --port 8000 --workers 8 --max-cache 1000 --shuffle-buffer 64 --min-width 256 --min-height 256 --min-person-height-ratio 0.25
```

如果 Hugging Face 提示 `laion/laion2B-en` 是 gated dataset，需要先提供 token：

```bash
HF_TOKEN=你的_token ./run_laion_people_api.sh
```

或者：

```bash
python laion_people_api.py --hf-token 你的_token
```

查看后台是否已经缓存到图：

```bash
curl http://127.0.0.1:8000/health
```

拿一张合格图：

```bash
curl "http://127.0.0.1:8000/api/image?timeout_seconds=10"
```

批量拿多张：

```bash
curl "http://127.0.0.1:8000/api/images?count=10&timeout_seconds=10"
```

返回里的 `download_url` 是本服务的本地图片链接，已经下载并通过检测，所以比直接返回 LAION 原始 URL 更可用。

如果 `/health` 里 `cached=0` 且 `checked=0`，通常是流式 shuffle 还在预热。可以进一步调小随机缓冲：

```bash
python laion_people_api.py --shuffle-buffer 64
```

如果你想更严格，只接受 OpenCV 人体检测通过的图：

```bash
python laion_people_api.py --strict-body-detection
```

默认模式会允许“检测到脸 + caption 有全身提示”的样本通过，这样速度和出图率会更好。

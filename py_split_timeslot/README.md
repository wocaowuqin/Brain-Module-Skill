# Python 拆分版（按 MATLAB 多文件风格）

这是把你当前的 Python 逻辑拆成“一个功能一个文件”的版本，不是 MATLAB 文件。

## 入口文件
- `main_generate_split.py`：生成 `phase1/phase3` 请求数据
- `main_generate_events_split.py`：生成并验证事件列表

## 主要模块
- `config.py`：常量配置
- `arrivals.py`：泊松到达
- `sampling.py`：寿命 / 带宽 / CPU / 内存采样
- `request_factory.py`：单个请求构造
- `grouping.py`：按时间槽分组
- `statistics_utils.py`：统计打印
- `request_generation.py`：请求生成主流程
- `event_generation.py`：事件生成主流程
- `vnf_catalog.py`：VNF 目录
- `data_generator_class.py`：类版本封装

## 用法
```bash
python main_generate_split.py
python main_generate_events_split.py
```

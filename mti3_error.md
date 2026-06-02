  1. 直接现象
     启动 dynamic_defense_dashboard/server.py 后，前端首页显示“请求失败”。
     实际失败的是：

  - /api/dashboard
  - /api/component-detail?section=muti3

  2. 根因
     server.py 默认把 muti3/Dataset/validata.csv 当成默认数据集，但你当前仓库里没有这个文件，只有：

  - validata2.csv
  - validata_sample.csv

  3. 为什么“有缓存还会失败”
     .runtime_cache/ 里虽然已经有：

  - muti3_traffic_detail.json
  - muti3_log_detail.json
  - muti3_graph_detail.json

  但这些缓存是旧路径下生成的，缓存指纹里记录的是类似：

  - /home/xiaowang/Desktop/Dynamic Defense/...

  而你现在的仓库路径是：

  - /home/xiaowang/Desktop/cph/dynamic-defense/...

  指纹校验过不了，后端就认为缓存“失效”，然后回退去读当前目录下的 validata.csv。
  由于这个文件不存在，接口直接 500。

  4. 还有一个放大问题
     代码里不止一处会触发这个回退：

  - build_multi3_section() 首页聚合会触发
  - get_section_detail(section="muti3") 详情页会触发

  所以表现上就是：

  - 首页失败
  - 点进 muti3 组件详情也失败

  5. 已做的修复思路
     修复不是改模型，也不是改前端文案，而是改缓存回退逻辑：

  - 当默认 validata.csv 不存在时：
      - 优先直接读取 .runtime_cache 里的已落盘 payload
      - 不再强依赖当前目录下必须存在 validata.csv
  - 同时把首页聚合和 muti3 详情页两条路径都补齐

  6. 修复后的行为
     现在即使当前仓库没有 validata.csv，只要 .runtime_cache 里有之前生成的 muti3_*_detail.json，页面也能正常显示缓存数据，不再报“请求失败”。
  7. 这个问题的本质教训
     这是一个“缓存设计依赖绝对路径指纹，但项目目录被重新 clone/移动后没有做兼容”的问题。
     所以只要你换目录、重新 clone，旧缓存很可能再次“看起来存在，但实际上不会被命中”。
